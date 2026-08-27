#!/usr/bin/env python3
"""vLLM OpenAI 호환 서버를 띄운다 (CUDA 전용).

run_llama_server.sh 의 vLLM 판이다. 다만 절반은 그대로 옮길 수 없다 —
llama.cpp 와 vLLM 은 메모리 모델이 정반대이기 때문이다.

  ── 사라진 것 ───────────────────────────────────────────────────────────────
  · 백엔드 선택(rocm/vulkan/metal)   → venv 하나. CUDA 전용
  · 라이브러리 사전점검 + ldd 복구    → 휠이라 .so 를 우리가 관리하지 않는다
  · probe_fit / probe_mem (약 350줄) → 아래 참조

    llama.cpp 는 -c 로 KV 를 "딱 그만큼" 잡으므로, 띄우기 전에 "컨텍스트를 얼마나
    잡을 수 있나" 를 알아내야 했다. 그래서 서버를 두 번 미리 띄워 물어보는 350줄이
    정당했다. vLLM 은 반대로 --gpu-memory-utilization 으로 풀을 **먼저 선점**하고
    남는 것을 전부 KV 블록으로 쓴다. 질문 자체가 뒤집힌다:

        llama.cpp : "컨텍스트를 얼마나 잡을 수 있나"  → 기동 전에 계산해야 한다
        vLLM      : "util 을 얼마로 주면 원하는 게 들어가나" → 기동 후 로그가 답한다

    그래서 사전 프로브 대신 §9 사후 파싱을 한다. vLLM 이 직접 찍어주는
    'GPU KV cache size: N tokens' 를 읽어 동시 수용 시퀀스 수를 보고한다.

  ── 그대로 통하는 것 ────────────────────────────────────────────────────────
  · "판단 기준과 화면 출력의 분리". 준비 판정은 /health 폴링으로(로그 문구가 바뀌어도
    안 깨진다), 화면에는 로그를 실시간으로 흘린다. 이 설계는 vLLM 에서도 유효하다.

  ── 새로 필요한 것 ──────────────────────────────────────────────────────────
  · util 자동 계산 (타 프로세스 점유를 빼고)
  · 툴 콜 파서 자동 결정 — 안 맞으면 조용히 텍스트로 샌다
  · 적재 가능성 사전 판정 — 부분 오프로드가 없으니 안 들어가면 수 분 뒤 OOM 이다
  · bench/runs.tsv 자동 기록 — 실측이 저절로 쌓이게

사용:
  ./run_vllm_server.py                        # models/ 를 훑어 대화형 선택
  ./run_vllm_server.py models/<이름>          # 지정
  ./run_vllm_server.py <HF리포>               # HF 에서 직접 (캐시로 받는다)
  ./run_vllm_server.py --port 8000 --util 0.9 --len 32768 --no-eager
  ./run_vllm_server.py --dry-run              # 실행할 명령만 보고 끝낸다
"""
import argparse
import ast
import datetime
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.environ.get("VENV_DIR", os.path.join(SCRIPT_DIR, ".venv"))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
BENCH_TSV = os.path.join(SCRIPT_DIR, "bench", "runs.tsv")

# download_model.py 의 판정 자산을 재사용한다. 같은 디렉터리에 나란히 있고,
# 양자화 판정 규칙이 두 벌로 갈리면 "받을 때는 ◎ 인데 띄울 때는 △" 같은 일이 생긴다.
sys.path.insert(0, SCRIPT_DIR)
try:
    import download_model as dm
except ImportError as exc:                                    # pragma: no cover
    print(f"download_model.py 를 찾지 못했습니다 ({exc}). 같은 디렉터리에 있어야 합니다.",
          file=sys.stderr)
    sys.exit(1)


def section(title):
    print("")
    print(f"── {title} ──")


# 대화형 여부는 한 곳에서만 판단한다. --yes 를 주거나 stdin 이 TTY 가 아니면 묻지 않는다.
# (스크립트·nohup·다른 프로젝트에서 부를 때 프롬프트에서 멈추면 안 된다)
_INTERACTIVE = True


def interactive():
    return _INTERACTIVE and sys.stdin.isatty()


def confirm(msg, default_yes, auto_reason=""):
    """예/아니오 확인. 비대화형이면 default_yes 를 그대로 쓴다."""
    if not interactive():
        print(f"{msg} → {'예' if default_yes else '아니오'} (자동{auto_reason})")
        return default_yes
    suffix = "(Y/n)" if default_yes else "(y/N)"
    a = (dm.prompt(f"{msg} {suffix}: ") or "").strip().lower()
    if not a:
        return default_yes
    return a.startswith("y")


def die(*lines, code=1):
    """stdout 을 비운 뒤 stderr 로 알리고 끝낸다.

    stdout 은 파이프/로그에서 블록 버퍼링이고 stderr 는 무버퍼라, 그냥 쓰면
    오류 메시지가 그 앞의 정상 출력보다 먼저 나가 원인과 맥락이 뒤집혀 보인다.
    나가는 문은 하나로 모아 두면 이 실수를 반복하지 않는다.
    """
    sys.stdout.flush()
    for ln in lines:
        print(ln, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(code)


# ─────────────────────────────────────────────────────────────────────────────
# 0. 인자
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv):
    p = argparse.ArgumentParser(
        description="vLLM 서버 기동 (CUDA 전용)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default=None,
                   help="모델 디렉터리 또는 HF 리포. 생략하면 models/ 에서 고른다")
    p.add_argument("--port", type=int, default=None,
                   help="지정하면 묻지 않는다. 생략하면 비어 있는 포트를 기본값으로 제안")
    p.add_argument("--host", default="127.0.0.1",
                   help="기본 127.0.0.1 — 외부 노출은 리버스 프록시로")
    p.add_argument("--util", type=float, default=None,
                   help="--gpu-memory-utilization. 생략하면 타 프로세스 점유를 빼고 자동 계산")
    p.add_argument("--len", dest="max_model_len", type=int, default=None,
                   help="--max-model-len. 생략하면 모델 기본값")
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--kv", dest="kv_cache_dtype", default="auto",
                   choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                   help="KV 캐시 양자화. 기본 auto(모델 dtype 그대로) — 안전한 쪽이 "
                        "기본이다. fp8 은 토큰 밀도 1.97배를 얻지만 조건이 붙는다: "
                        "슬라이딩 윈도우 층이 많으면 vLLM 이 권하지 않고, SM80 의 "
                        "Triton 백엔드(Gemma4 등 head_dim 512)는 아예 거부한다. "
                        "docs/quantization.md §5 · docs/performance.md §5")
    p.add_argument("--eager", dest="eager", action="store_true", default=True,
                   help="--enforce-eager (기본 켜짐). CUDA 그래프 캡처를 건너뛰어 기동이 빠르다")
    p.add_argument("--no-eager", dest="eager", action="store_false",
                   help="CUDA 그래프를 캡처한다. 기동은 느려지고 추론은 빨라진다")
    p.add_argument("--tool-parser", default=None,
                   help="툴 콜 파서를 직접 지정 (자동 추정을 무시)")
    p.add_argument("--no-tools", action="store_true",
                   help="툴 콜링을 켜지 않는다")
    p.add_argument("--reasoning-parser", default=None,
                   help="reasoning 파서를 직접 지정 (자동 추정 무시)")
    p.add_argument("--no-reasoning", action="store_true",
                   help="reasoning 파서를 붙이지 않는다")
    p.add_argument("--name", default=None,
                   help="--served-model-name. 생략하면 디렉터리 이름을 쓴다")
    p.add_argument("--timeout", type=int, default=180,
                   help="준비 대기 상한(초, 기본 180). 넘겨도 서버는 계속 로딩한다")
    p.add_argument("--dry-run", action="store_true",
                   help="실행할 명령만 출력하고 끝낸다")
    p.add_argument("--kill-existing", dest="kill_existing", action="store_true", default=None,
                   help="기존 vLLM 을 묻지 않고 종료한다")
    p.add_argument("--keep-existing", dest="kill_existing", action="store_false",
                   help="기존 vLLM 을 묻지 않고 그대로 둔다 (다중 인스턴스)")
    p.add_argument("--force", action="store_true",
                   help="--kill-existing 의 별칭 (하위 호환)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="모든 확인을 자동 승인하고 아무것도 묻지 않는다. "
                        "다른 프로젝트에서 벤치마크용으로 부를 때 쓴다")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="이후 인자를 vllm serve 에 그대로 넘긴다. "
                        "⚠ REMAINDER 라 **뒤에 오는 것을 전부 삼킨다** — 반드시 맨 끝에 둘 것. "
                        "예: ./run_vllm_server.py <model> --util 0.5 --extra --kv-cache-dtype fp8")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# 1. venv
# ─────────────────────────────────────────────────────────────────────────────
def find_vllm():
    exe = os.path.join(VENV_DIR, "bin", "vllm")
    if os.path.isfile(exe) and os.access(exe, os.X_OK):
        return exe
    exe = shutil.which("vllm")
    if exe:
        print(f"  ⚠ {VENV_DIR} 에 vllm 이 없어 PATH 의 것을 씁니다: {exe}")
        return exe
    die("vllm 을 찾지 못했습니다. 먼저 환경을 만드십시오:",
        "  ./setup_vllm.sh")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPU
# ─────────────────────────────────────────────────────────────────────────────
def gpu_info():
    """(이름, cc, 라벨 VRAM MiB, 타 프로세스 점유 MiB). 없으면 None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        name, cap, total = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None
    used = 0
    try:
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        for line in a.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                used += int(parts[1])
    except Exception:
        pass
    return name, float(cap), int(total), used


def physical_vram_mib(name, label_mib):
    """실사용 상한. 대개 라벨과 같지만 CMP 170HX 는 다르다.

    nvidia-smi 는 65536 MiB 를 보고하지만 cmpunlocker 의 clamp 가 존재하지 않는
    구간을 PMA 에서 제외하므로 실제로 쓸 수 있는 것은 약 63.4 GiB 다.
    (util 의 **분모**는 여전히 라벨값이라는 점에 주의 — 아래 계산 참조.)
    """
    env = os.environ.get("PHYS_VRAM_MIB")
    if env and env.isdigit():
        return int(env)
    if "CMP 170HX" in name:
        return 64921
    return label_mib


# ─────────────────────────────────────────────────────────────────────────────
# 3. 모델 선택
# ─────────────────────────────────────────────────────────────────────────────
def local_models():
    """models/ 아래에서 vLLM 이 읽을 수 있는 디렉터리를 찾는다.

    판별자는 config.json 이다. GGUF 만 든 디렉터리는 자연히 걸러진다 —
    별도의 마커 파일을 만들 이유가 없다.
    """
    out = []
    for cfg in sorted(glob.glob(os.path.join(MODELS_DIR, "*", "config.json"))):
        d = os.path.dirname(cfg)
        size = 0
        for root, _dirs, names in os.walk(d):
            for n in names:
                try:
                    size += os.path.getsize(os.path.join(root, n))
                except OSError:
                    pass
        out.append((d, size))
    return out


def choose_model():
    models = local_models()
    if not models:
        die(f"{MODELS_DIR}/ 에 vLLM 이 읽을 모델이 없습니다 "
            f"(config.json 이 있는 디렉터리를 찾습니다).",
            "먼저 받으십시오:  python3 download_model.py <HF저장소 URL>")
    section("모델 선택")
    for i, (d, size) in enumerate(models, 1):
        print(f"  [{i}] {os.path.basename(d):<48} {dm._fmt_bytes(size)}")
    if not interactive():
        print(f"  (묻지 않음 — [1] 선택). 지정하려면 인자로 모델 경로를 주십시오.")
        return models[0][0]
    while True:
        a = dm.prompt("번호 선택 [기본값 1]: ") or "1"
        if a.isdigit() and 1 <= int(a) <= len(models):
            return models[int(a) - 1][0]
        print("올바른 번호를 선택하세요.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 모델 판정
#
# download_model.py 의 gate_repo 와 같은 일을 로컬 디렉터리에 대해 한다. 받을 때
# 이미 한 번 걸렀더라도 여기서 다시 보는 이유는, 사람이 손으로 받아 넣은 디렉터리나
# HF 리포를 직접 지정한 경우에는 그 게이트를 거치지 않았기 때문이다.
# ─────────────────────────────────────────────────────────────────────────────
def inspect_model(path, gpu):
    """(config, 가중치 바이트, 양자화라벨). 로컬이 아니면 크기는 0."""
    section("모델 판정")
    is_local = os.path.isdir(path)
    print(f"  대상       : {path}{'' if is_local else '  (HF 리포 — 캐시로 받습니다)'}")

    config = {}
    if is_local:
        try:
            with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
                config = json.load(f)
        except Exception as exc:
            print(f"  ⚠ config.json 을 읽지 못했습니다 ({exc})")
    else:
        # HF 리포는 아직 로컬에 없다. 판정만 원격으로 한 번 본다.
        owner, repo, rev = dm.hf_repo_from_input(f"https://huggingface.co/{path}")
        if owner:
            config = dm._fetch_json(dm.hf_file_url(owner, repo, rev, "config.json")) or {}

    archs = config.get("architectures") or []
    if archs:
        print(f"  아키텍처   : {', '.join(archs)}")

    quant = "none"
    qc = config.get("quantization_config") or {}
    method = (qc.get("quant_method") or "").lower()
    if method:
        mark, desc = dm.QUANT_VERDICT.get(method, ("?", "vLLM 지원 여부 미상"))
        fbits = dm._float_quant_bits(method, qc, os.path.basename(path))
        quant = method
        if fbits:
            quant = f"{method}-fp{fbits}"
            cap = gpu[1] if gpu else 0.0
            need = dm.FLOAT_QUANT_MIN_CAP[fbits]
            gen = dm.FLOAT_QUANT_GEN[fbits]
            if cap and cap >= need:
                mark, desc = "◎", f"fp{fbits} 텐서코어 있음"
            else:
                # vLLM 이 Marlin 가중치 전용 경로로 자동 폴백한다. 기동은 된다.
                mark = "△"
                desc = (f"fp{fbits} 텐서코어 없음 → Marlin 가중치 전용 폴백 "
                        f"(VRAM·대역폭만 이득, 연산 이득 없음)")
        print(f"  양자화     : {mark} {quant} — {desc}")
        if mark == "✗":
            sys.stdout.flush()      # stdout 은 파이프에서 블록 버퍼링, stderr 는 무버퍼
            print("  ⛔ 이 양자화는 이 GPU 에서 동작하지 않습니다. 기동해도 실패합니다.",
                  file=sys.stderr)
            sys.stderr.flush()
            if not confirm("     그래도 계속할까요?", default_yes=False,
                           auto_reason=" — 중단합니다"):
                sys.exit(1)
    else:
        dt = config.get("torch_dtype") or config.get("dtype") or "?"
        quant = str(dt)
        print(f"  양자화     : ○ 없음 (원본 {dt})")

    # ── 적재 가능성 ── vLLM 에는 -ngl 이 없다. 안 들어가면 수 분 기다렸다 OOM 이다.
    weight_b = 0
    if is_local:
        for root, _d, names in os.walk(path):
            for n in names:
                if n.lower().endswith((".safetensors", ".bin", ".pth")):
                    try:
                        weight_b += os.path.getsize(os.path.join(root, n))
                    except OSError:
                        pass
        if weight_b:
            print(f"  가중치     : {dm._fmt_bytes(weight_b)}")
    return config, weight_b, quant


# ─────────────────────────────────────────────────────────────────────────────
# 5. util / 컨텍스트
# ─────────────────────────────────────────────────────────────────────────────
def decide_util(gpu, weight_b, want):
    """--gpu-memory-utilization 을 정한다.

    함정: util 은 '남은 VRAM' 이 아니라 **GPU 전체** 대비 비율이다. llama-server 가
    19GB 를 쥔 채로 vLLM 에 0.9 를 주면 합이 넘친다. 그래서 타 프로세스 점유를
    실제로 읽어서 빼고 계산한다.
    """
    section("--gpu-memory-utilization")
    if not gpu:
        print("  nvidia-smi 없음 — 자동 계산 불가")
        return want if want is not None else 0.9
    name, cap, label_mib, other_mib = gpu
    phys_mib = physical_vram_mib(name, label_mib)

    avail_mib = max(0, phys_mib - other_mib)
    print(f"  VRAM 라벨    : {label_mib} MiB   ← util 의 분모")
    if phys_mib != label_mib:
        print(f"  물리 상한    : {phys_mib} MiB   ← 라벨보다 작습니다 (clamp)")
    print(f"  타 프로세스  : {other_mib} MiB")
    print(f"  vLLM 가용    : {avail_mib} MiB")

    ceiling = avail_mib / label_mib
    auto = min(0.95, max(0.10, ceiling - 0.05))     # 단편화·CUDA 컨텍스트 여유 5%
    print(f"  상한 util    : {ceiling:.2f}")
    if want is not None:
        if want > ceiling:
            print(f"  ⚠ 지정한 util {want:.2f} 가 상한 {ceiling:.2f} 를 넘습니다 — OOM 가능")
        print(f"  → 사용       : {want:.2f} (지정)")
        _check_budget(want, label_mib, weight_b)
        return want
    chosen = ask_util(auto, ceiling, label_mib, weight_b)
    print(f"  → 사용       : {chosen:.2f}")
    _check_budget(chosen, label_mib, weight_b)
    return chosen


def ask_util(auto, ceiling, label_mib, weight_b):
    """util 을 대화형으로 고르게 한다. 모델 선택과 같은 결의 UI.

    util 은 이 박스에서 가장 자주 바꾸는 손잡이다 — ComfyUI 와 자리를 나눠 쓸지,
    vLLM 단독으로 쓸지에 따라 매번 달라진다. 모델은 물어보면서 이건 안 물어보면
    매번 --util 을 타이핑해야 한다.

    비대화형(스크립트·nohup)이면 묻지 않고 자동값을 쓴다.
    """
    if not interactive():
        print(f"  (묻지 않음 — 자동값 {auto:.2f} 사용)")
        return auto

    w_mib = weight_b / (1024 * 1024) if weight_b else 0

    def head(u):
        """이 util 이 무엇을 뜻하는지 한 줄로."""
        budget = u * label_mib
        if not w_mib:
            return f"{budget:.0f} MiB"
        kv = budget - w_mib
        free = label_mib - budget
        s = f"{budget:>6.0f} MiB (KV 여유 {kv:>6.0f})"
        if kv < 0:
            return s + "  ⛔ 가중치보다 작다"
        return s + f", 남는 VRAM {free:.0f}"

    opts = [(auto, "자동 — vLLM 단독")]
    for u, why in ((0.50, "ComfyUI 등과 공존"), (0.35, "가벼운 실험")):
        if u < auto - 0.02:
            opts.append((u, why))

    print("")
    for i, (u, why) in enumerate(opts, 1):
        print(f"    [{i}] {u:.2f}  {head(u):<44} {why}")
    print(f"    직접 입력도 됩니다 (예: 0.7)")

    while True:
        a = dm.prompt(f"  선택 [기본값 1]: ") or "1"
        if a.isdigit() and 1 <= int(a) <= len(opts):
            return opts[int(a) - 1][0]
        try:
            v = float(a)
        except ValueError:
            print("    번호 또는 0~1 사이 숫자를 넣으십시오."); continue
        if not (0 < v <= 1):
            print("    util 은 0 초과 1 이하입니다."); continue
        if v > ceiling:
            print(f"    ⚠ 상한 {ceiling:.2f} 를 넘습니다 — OOM 가능")
        return v


def _check_budget(util, label_mib, weight_b):
    """util 이 정해진 뒤, 그 예산 안에 가중치가 들어가는지 본다.

    vLLM 은 가중치를 다 올리고 나서야 'Available KV cache memory: -13.6 GiB' 로 죽는다.
    그때까지 20초 넘게 기다려야 하므로, 계산으로 미리 아는 편이 낫다.
    """
    if not weight_b:
        return
    budget_mib = util * label_mib
    weight_mib = weight_b / (1024 * 1024)
    head_mib = budget_mib - weight_mib
    print(f"  KV 여유 추정 : 약 {head_mib:.0f} MiB "
          f"(예산 {budget_mib:.0f} − 가중치 {weight_mib:.0f})")
    if head_mib < 0:
        sys.stdout.flush()
        print(f"  ⛔ 예산({budget_mib:.0f} MiB)이 가중치({weight_mib:.0f} MiB)보다 작습니다. "
              f"기동하면 KV 캐시가 음수가 되어 엔진이 죽습니다.", file=sys.stderr)
        print("     GPU를 쓰는 다른 프로세스를 내리거나 --util 을 직접 지정하십시오.",
              file=sys.stderr)
        sys.stderr.flush()
        if not confirm("     그래도 진행할까요?", default_yes=False,
                       auto_reason=" — 중단합니다"):
            sys.exit(1)
    elif head_mib < 2048:
        print("  ⚠ KV 여유가 2 GiB 미만입니다. 컨텍스트를 짧게 잡으십시오.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 툴 콜 파서
#
# 문서의 '최대 사고 지점'. 파서가 안 맞으면 오류가 나는 게 아니라 tool_calls 가
# 조용히 비어 오고 도구 호출이 본문 텍스트로 새어나온다. 그래서 모르면 추측해서
# 붙이지 않고 비워 둔다 — 최종 판정은 실측(10회 중 9회)이지 아래 표가 아니다.
# ─────────────────────────────────────────────────────────────────────────────
def available_parsers(kind="tool"):
    """설치된 vLLM 이 실제로 아는 파서 이름. 하드코딩하지 않는다.

    kind="tool" → vllm/tool_parsers/__init__.py 의 _TOOL_PARSERS_TO_REGISTER
    kind="reasoning" → vllm/reasoning/__init__.py 의 _REASONING_PARSERS_TO_REGISTER

    vllm 을 import 하면 정확하지만 전체를 끌어와 수 초가 걸린다. 등록표가 평범한
    dict 리터럴이라 ast 로 1ms 만에 읽을 수 있다. vLLM 을 올려도 자동으로 따라간다.
    """
    pkg, var = (("tool_parsers", "_TOOL_PARSERS_TO_REGISTER") if kind == "tool"
                else ("reasoning", "_REASONING_PARSERS_TO_REGISTER"))
    pats = glob.glob(os.path.join(VENV_DIR, "lib", "python3.*", "site-packages",
                                  "vllm", pkg, "__init__.py"))
    for path in pats:
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if var in names and isinstance(node.value, ast.Dict):
                out = {k.value for k in node.value.keys
                       if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if out:
                    return out
    return set()


# 아키텍처/이름 → 파서 후보. 위에서부터 먼저 맞는 것을 쓴다.
PARSER_RULES = [
    (r"gemma_?4|gemma4",                       ["gemma4", "functiongemma"]),
    (r"qwen3.*coder",                          ["qwen3_coder"]),
    (r"qwen3|qwen_?3",                         ["qwen3_xml", "hermes"]),
    (r"qwen2|qwen_?2",                         ["hermes"]),
    (r"llama.?4",                              ["llama4_pythonic", "llama4_json"]),
    (r"llama.?3|llama",                        ["llama3_json"]),
    (r"mistral|mixtral|ministral",             ["mistral"]),
    (r"deepseek.?v4",                          ["deepseek_v4"]),
    (r"deepseek.?v3",                          ["deepseek_v3"]),
    (r"glm.?4\.?7|glm47",                      ["glm47"]),
    (r"glm.?4",                                ["glm45"]),
    (r"granite.?4",                            ["granite4"]),
    (r"granite",                               ["granite"]),
    (r"internlm",                              ["internlm"]),
    (r"jamba",                                 ["jamba"]),
    (r"kimi.?k3",                              ["kimi_k3"]),
    (r"kimi",                                  ["kimi_k2"]),
    (r"minimax",                               ["minimax_m2"]),
    (r"phi.?4",                                ["phi4_mini_json"]),
    (r"seed.?oss",                             ["seed_oss"]),
    (r"hunyuan",                               ["hunyuan_a13b"]),
    (r"olmo",                                  ["olmo3"]),
    (r"ernie",                                 ["ernie45"]),
    (r"step3",                                 ["step3"]),
    (r"cohere|command",                        ["cohere_command4", "cohere_command3"]),
]


# 아키텍처/이름 → reasoning 파서. 사고 과정을 본문에서 분리하지 않으면
# 내부 독백이 그대로 사용자에게 나간다 (실측으로 확인했다).
REASONING_RULES = [
    (r"gemma_?4|gemma4",           ["gemma4"]),
    (r"qwen3|qwen_?3",             ["qwen3"]),
    (r"deepseek.?v4",              ["deepseek_v4"]),
    (r"deepseek.?v3",              ["deepseek_v3"]),
    (r"deepseek.?r1|r1.?distill",  ["deepseek_r1"]),
    (r"glm.?4\.?7|glm47",          ["glm47"]),
    (r"glm.?4",                    ["glm45"]),
    (r"granite",                   ["granite"]),
    (r"minimax",                   ["minimax_m2"]),
    (r"kimi.?k3",                  ["kimi_k3"]),
    (r"kimi",                      ["kimi_k2"]),
    (r"seed.?oss",                 ["seed_oss"]),
    (r"olmo",                      ["olmo3"]),
    (r"ernie",                     ["ernie45"]),
    (r"hunyuan",                   ["hunyuan_a13b"]),
    (r"mistral|magistral",         ["mistral"]),
    (r"step3",                     ["step3"]),
]


def decide_reasoning_parser(path, config, override, disabled):
    """사고 과정을 reasoning_content 로 분리할 파서.

    실측 사고: Qwen3.8 을 파서 없이 띄웠더니 "Thinking Process: 1. Analyze the
    Request..." 라는 영어 내부 독백이 **본문(content)에 그대로** 나왔다.
    한국어로 물었는데 영어 사고가 본문에 섞여 나오므로 파이프라인에서 그대로 쓸 수 없다.
    """
    if disabled:
        return None
    known = available_parsers("reasoning")
    if override:
        if known and override not in known:
            print(f"  ⚠ '{override}' 는 이 vLLM 이 아는 reasoning 파서가 아닙니다")
        print(f"  reasoning  : {override} (지정)")
        return override
    hay = " ".join([os.path.basename(os.path.abspath(path))] +
                   (config.get("architectures") or [])).lower()
    for pattern, cands in REASONING_RULES:
        if re.search(pattern, hay):
            for c in cands:
                if not known or c in known:
                    print(f"  reasoning  : {c} (추정) — 사고 과정을 "
                          f"reasoning_content 로 분리합니다")
                    return c
    print(f"  reasoning  : 없음 — 사고형 모델이면 내부 독백이 본문에 섞일 수 있습니다")
    return None


def decide_tool_parser(path, config, override, disabled):
    section("툴 콜링")
    if disabled:
        print("  --no-tools — 켜지 않습니다")
        return None
    known = available_parsers()
    if override:
        if known and override not in known:
            print(f"  ⚠ '{override}' 는 이 vLLM 이 아는 파서가 아닙니다 "
                  f"(알려진 것 {len(known)}종). 그대로 넘깁니다.")
        print(f"  파서       : {override} (지정)")
        return override

    hay = " ".join([os.path.basename(os.path.abspath(path))] +
                   (config.get("architectures") or [])).lower()
    for pattern, cands in PARSER_RULES:
        if re.search(pattern, hay):
            for c in cands:
                if not known or c in known:
                    print(f"  파서       : {c} (이름/아키텍처에서 추정)")
                    print("  ⚠ 추정입니다. tools 를 준 요청으로 tool_calls 가 실제로 파싱되는지")
                    print("     확인하십시오 — 안 맞으면 오류 없이 본문 텍스트로 샙니다.")
                    return c
    print(f"  파서       : 결정하지 못했습니다 (알려진 {len(known)}종 중 매칭 없음)")
    print("     추측해서 붙이면 조용히 틀린 결과가 나오므로 비워 둡니다.")
    print("     직접 주려면: --tool-parser <이름>")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. 기동
# ─────────────────────────────────────────────────────────────────────────────
def running_vllm():
    """지금 떠 있거나 **기동 중인** vLLM 을 모두 찾는다.

    ★ 포트만 봐서는 못 잡는다. vLLM 은 파이썬 임포트 → 가중치 적재 → KV 생성 을
    다 마친 **1~2분 뒤에야 포트를 연다.** 그 사이에 두 번째를 띄우면 포트가 비어
    있으니 충돌이 감지되지 않고, 둘 다 진행하다가 뒤엣놈이 이렇게 죽는다:

      ValueError: Free memory on device cuda:0 (44.86/63.39 GiB) on startup is
      less than desired GPU memory utilization (0.94, 59.59 GiB)

    실제로 그 사고가 났다. 그래서 포트가 아니라 **프로세스**를 본다.
    """
    out = []
    try:
        r = subprocess.run(["pgrep", "-af", "bin/vllm serve"],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return out
    me = str(os.getpid())
    for line in r.stdout.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid == me or "pgrep" in cmd:
            continue
        port = model = None
        toks = cmd.split()
        for i, t in enumerate(toks):
            if t == "--port" and i + 1 < len(toks):
                port = toks[i + 1]
            elif t == "--served-model-name" and i + 1 < len(toks):
                model = toks[i + 1]
        if model is None:
            for t in toks:
                if "/models/" in t:
                    model = os.path.basename(t.rstrip("/"))
                    break
        out.append((pid, model or "?", port or "?"))
    return out


def gpu_mib_of(pid):
    """이 vLLM 인스턴스가 잡은 VRAM.

    ★ pid 를 그대로 nvidia-smi 와 대조하면 안 된다. `vllm serve` 의 최상위 프로세스는
    APIServer 이고, **GPU 를 실제로 잡는 것은 그 자식인 EngineCore** 다.
    (실측: APIServer 2023436 → 자식 EngineCore 2025442 가 60,272 MiB 보유)
    그래서 자손까지 훑어 합산한다.
    """
    try:
        kids = subprocess.run(["pgrep", "-P", str(pid)],
                              capture_output=True, text=True, timeout=10).stdout.split()
    except Exception:
        kids = []
    family = {str(pid), *kids}
    for k in list(kids):                       # 손자까지 (EngineCore 가 더 낳는 경우)
        try:
            family.update(subprocess.run(["pgrep", "-P", k], capture_output=True,
                                         text=True, timeout=10).stdout.split())
        except Exception:
            pass
    total = 0
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            a, _, b = line.partition(",")
            if a.strip() in family:
                total += int(b.strip())
    except Exception:
        return None
    return total or None


def handle_existing(kill_existing):
    """기존 vLLM 을 어떻게 할지 정한다. (종료했나 bool)

    **여러 인스턴스를 띄우는 것 자체는 정상 사용이다** — 다른 모델을 동시에 쓰거나
    A/B 비교를 할 때 그렇게 한다. (vLLM 한 프로세스는 모델 하나만 서빙한다.
    `--served-model-name` 은 별칭일 뿐이고, `--lora-modules` 는 같은 베이스의
    어댑터만 가능하다.) 그래서 죽이지 않고 **포트를 바꿔 공존**하는 길을 연다.
    """
    procs = running_vllm()
    if not procs:
        return False
    section("기존 vLLM")
    for pid, model, port in procs:
        mib = gpu_mib_of(pid)
        print(f"  pid {pid:<8} {model:<46} :{port}"
              + (f"  {mib} MiB" if mib else "  (기동 중 — GPU 미점유)"))
    if kill_existing is None:
        kill_existing = confirm("  이 vLLM 을 종료할까요?", default_yes=False,
                                auto_reason=" — 그대로 두고 다른 포트를 씁니다")
    if not kill_existing:
        print("  그대로 둡니다. 포트가 겹치지 않게 고르십시오.")
        return False
    # 프로세스 그룹째 보낸다. APIServer 만 죽이면 GPU 를 쥔 자식 EngineCore 가
    # 살아남아 VRAM 이 안 풀리는 일이 생긴다 (run_vllm_server 는 start_new_session
    # 으로 띄우므로 pid 가 곧 그룹 리더다).
    def _sig(pid, sig):
        for target in (lambda: os.killpg(int(pid), sig), lambda: os.kill(int(pid), sig)):
            try:
                target(); return
            except OSError:
                continue

    for pid, _, _ in procs:
        _sig(pid, signal.SIGTERM)
    for _ in range(40):
        if not running_vllm():
            break
        time.sleep(0.5)
    for pid, _, _ in running_vllm():
        _sig(pid, signal.SIGKILL)
    time.sleep(1)
    print("  종료했습니다.")
    wait_vram_release()
    return True


def pick_port(want, host):
    """포트를 정한다. --port 를 주면 묻지 않는다.

    기존 vLLM 이 있든 없든 **항상** 물어본다 — 다중 인스턴스가 정상 사용이므로
    사용자가 매번 의식적으로 고르는 편이 안전하다.
    """
    if want is not None:
        if port_in_use(host, want):
            print(f"  ⚠ 포트 {want} 가 이미 사용 중입니다 (지정값이라 그대로 진행)")
        return want
    base = 8000
    while port_in_use(host, base) and base < 8100:
        base += 1
    if not interactive():
        print(f"  포트 {base} (자동 — 비어 있는 첫 포트)")
        return base
    used = [p for p in (8000, 8001, 8002, 8003) if port_in_use(host, p)]
    if used:
        print(f"  사용 중: {', '.join(str(x) for x in used)}")
    while True:
        a = (dm.prompt(f"  포트 [기본값 {base}]: ") or str(base)).strip()
        if not a.isdigit():
            print("    숫자를 넣으십시오."); continue
        v = int(a)
        if not (1 <= v <= 65535):
            print("    1~65535 범위여야 합니다."); continue
        if port_in_use(host, v):
            if not confirm(f"    포트 {v} 는 이미 사용 중입니다. 그래도 쓸까요?",
                           default_yes=False):
                continue
        return v


def port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def wait_vram_release(timeout=30):
    """포트가 닫혀도 VRAM 반환은 몇 초 늦다. 안정될 때까지 기다린다.

    실측 사고: 이걸 안 하고 곧바로 GPU를 재면 죽은 서버가 쥐고 있던 60GB가 그대로
    보여서 util 이 0.10 으로 계산됐고, 18.2GiB 모델에 6.4GiB만 주는 바람에
    'Available KV cache memory: -13.6 GiB' 로 엔진이 죽었다.
    """
    g = gpu_info()
    if not g:
        return
    prev = g[3]
    if prev == 0:
        return
    print(f"     VRAM 반환 대기 (현재 타 프로세스 {prev} MiB)...", end="", flush=True)
    stable = 0
    for _ in range(int(timeout / 0.5)):
        time.sleep(0.5)
        g = gpu_info()
        cur = g[3] if g else 0
        if cur == prev:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
            prev = cur
        if cur == 0:
            break
    print(f" {prev} MiB")


def child_env():
    """vllm 자식 프로세스에 넘길 환경.

    실측 사고: vllm 을 절대경로로 띄우면 프로세스는 뜨지만 **PATH 에 .venv/bin 이 없다.**
    그래서 FlashInfer 가 샘플링 커널을 JIT 컴파일하려고 `ninja` 를 subprocess 로 부를 때
    `FileNotFoundError: 'ninja'` 로 엔진이 죽었다 — ninja 는 .venv/bin 에 멀쩡히 있는데도.

    venv 를 activate 하고 실행하면 안 나는 문제라 재현 조건이 헷갈린다. 여기서 못박는다:
      - PATH 앞에 .venv/bin (ninja 등 venv 스크립트)
      - PATH 앞에 nvidia/cu*/bin (nvcc — JIT 컴파일에 필요)
      - CUDA_HOME 미설정 시 nvidia/cu* 를 가리킨다 (flashinfer 가 이 순서로 찾는다)
    """
    env = dict(os.environ)

    # conda 를 자식 PATH 에서 뺀다.
    #
    # 실측 사고: JIT 컴파일이 conda 의 툴체인을 집어 링크에서 죽었다 —
    #   /home/lota/miniconda3/.../x86_64-conda-linux-gnu/bin/ld: cannot find -lcuda
    # libcuda.so 는 /usr/lib/x86_64-linux-gnu 에 멀쩡히 있는데, conda 링커는
    # 자기 sysroot 만 보므로 시스템 드라이버 라이브러리를 못 찾는다.
    # setup_vllm.sh §0 이 같은 이유로 같은 일을 한다.
    conda_roots = []
    for v in ("CONDA_PREFIX", "CONDA_EXE"):
        val = env.get(v)
        if val:
            conda_roots.append(val if v == "CONDA_PREFIX" else val[: val.rfind("/bin/")])
    for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        r = os.path.join(os.path.expanduser("~"), d)
        if os.path.isdir(r):
            conda_roots.append(r)
    if conda_roots:
        kept = [x for x in env.get("PATH", "").split(os.pathsep)
                if x and not any(x == r or x.startswith(r + "/") for r in conda_roots)]
        env["PATH"] = os.pathsep.join(kept) or "/usr/local/bin:/usr/bin:/bin"
        for v in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONPATH"):
            env.pop(v, None)

    # 링커가 드라이버 라이브러리(libcuda.so)를 찾게 해준다. 드라이버가 주는 것이라
    # CUDA pip 패키지에는 없다.
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        if os.path.exists(os.path.join(d, "libcuda.so")):
            env["LIBRARY_PATH"] = os.pathsep.join(
                [d] + ([env["LIBRARY_PATH"]] if env.get("LIBRARY_PATH") else []))
            break

    extra = [os.path.join(VENV_DIR, "bin")]
    cuda_root = ""
    for d in sorted(glob.glob(os.path.join(
            VENV_DIR, "lib", "python3.*", "site-packages", "nvidia", "cu*"))):
        if os.path.isfile(os.path.join(d, "bin", "nvcc")):
            extra.append(os.path.join(d, "bin"))
            cuda_root = d
            break
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    if cuda_root and not env.get("CUDA_HOME") and not env.get("CUDA_PATH"):
        env["CUDA_HOME"] = cuda_root

    # FlashInfer 샘플러 JIT 를 기본으로 끈다.
    #
    # 실측: ninja 를 찾게 해줬더니 이번엔 nvcc 가 컴파일에 실패했다 —
    #   error "CUDA compiler and CUDA toolkit headers are incompatible"
    # flashinfer 0.6.x 가 번들한 CCCL 헤더와 이 환경의 CUDA 13.3 이 안 맞는다.
    # 샘플러는 **최적화이지 필수가 아니다**(없으면 PyTorch 네이티브 경로로 간다).
    # 기동 때마다 커널을 JIT 컴파일하는 것 자체가 깨지기 쉬운 의존이라 기본으로 끈다.
    # 되살리려면 VLLM_USE_FLASHINFER_SAMPLER=1 로 실행한다.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env, extra, cuda_root


def format_cmd(cmd):
    """사람이 그대로 복사해 붙일 수 있게 --플래그와 그 값을 한 줄에 묶는다.

    토큰마다 줄을 바꾸면 '--port' 와 '8000' 이 따로 놀아서 읽기도 고치기도 나쁘다.
    """
    lines, i = [], 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            lines.append(f"{tok} {cmd[i + 1]}")
            i += 2
        else:
            lines.append(tok)
            i += 1
    head = " ".join(lines[:3])          # <vllm> serve <model> 은 한 줄에
    rest = lines[3:]
    out = ["  " + head] + [f"    {x}" for x in rest]
    return " \\\n".join(out)


def _fp8_kv_worth_it(config):
    """fp8 KV 가 이 모델에서 이득일 만한가.

    ★ 슬라이딩 윈도우 층이 많으면 아니다. 그 층들의 KV 는 창 크기에서 잘리므로
      길이에 비례해 자라지 않는다 — 절반으로 줄일 대상 자체가 작다.
      vLLM 공식 평가도 "슬라이딩 윈도우 레이어가 많을 때" 를 fp8 KV 비권장
      조건으로 명시한다 (2026-04-22 블로그).
      게다가 SM80 에서 head_dim > 256 인 모델(Gemma4 의 full_attention 512)은
      FlashAttention 이 못 받아 Triton 으로 폴백하고, Triton 은 fp8 KV 에
      SM89+ 를 요구해 **기동 자체가 실패한다** (실측).
    """
    if not isinstance(config, dict):
        return True
    t = config.get("text_config") or config
    lt = t.get("layer_types") or []
    if lt:
        sw = sum(1 for x in lt if "sliding" in str(x).lower())
        if sw > len(lt) / 2:
            return False
    for k in ("head_dim", "global_head_dim"):
        v = t.get(k)
        if isinstance(v, int) and v > 256:
            return False
    return True


def build_cmd(vllm_bin, model, args, util, parser, reasoning=None):
    # --served-model-name 을 반드시 준다. 안 주면 vLLM 이 **전체 경로**를 모델 id 로
    # 쓰기 때문에 API 호출 때마다 절대경로를 적어야 한다 (실측으로 확인했다).
    served = args.name or os.path.basename(os.path.abspath(model.rstrip("/")))
    cmd = [vllm_bin, "serve", model,
           "--served-model-name", served,
           "--host", args.host, "--port", str(args.port),
           "--gpu-memory-utilization", f"{util:.2f}"]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.max_num_seqs:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.eager:
        cmd += ["--enforce-eager"]
    # --extra 로 직접 지정했으면 그쪽을 존중한다 (중복 지정은 vLLM 이 거부한다)
    if not any(a.startswith("--kv-cache-dtype") for a in args.extra):
        cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
    if parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    if reasoning:
        cmd += ["--reasoning-parser", reasoning]
    cmd += [a for a in args.extra if a != "--"]
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# 8. 준비 대기
#
# run_llama_server.sh 의 설계를 그대로 가져온다: **판단은 /health, 출력은 로그**.
# 로그 문구로 준비를 판정하면 버전업 때마다 깨지지만, 화면에는 로그가 흘러야
# 무슨 일이 일어나는지 보인다. 둘을 분리하면 양쪽을 다 얻는다.
#
# 다만 타임아웃은 짧게 잡는다(기본 180초). 긴 타임아웃은 정상 경로에서 아무 이득이
# 없고 비정상 경로에서만 작동하는데, 비정상일 때야말로 빨리 알아야 하기 때문이다.
# 넘겨도 서버는 죽이지 않는다 — 계속 로딩 중일 수 있으므로 관찰 방법만 알려준다.
# ─────────────────────────────────────────────────────────────────────────────
# 치명 패턴은 **두 단계**로 판정한다.
#
# 실측 사고: `trust_remote_code` 만 보고 죽였더니, vLLM 이 기동 시 찍는 정상 INFO 줄
#   "Initializing a V1 LLM engine ... trust_remote_code=False, dtype=torch.bfloat16, ..."
# 에 걸려서 **멀쩡히 로딩 중인 서버를 종료시켰다.** 엔진 설정 덤프에는 거의 모든 옵션
# 이름이 문자열로 들어있으므로, 옵션 이름을 패턴으로 쓰면 반드시 오탐이 난다.
#
#   HARD : 그 문구 자체가 오류인 것. 어느 줄에 있든 치명으로 본다
#   SOFT : 정상 로그에도 나올 수 있는 것. **오류 표지가 있는 줄에서만** 인정한다
ERROR_MARK = re.compile(r"\bERROR\b|\bCRITICAL\b|Traceback \(most recent|"
                        r"^\s*raise |Error:|\bException\b")

FATAL_HARD = [
    (r"CUDA out of memory|OutOfMemoryError",
     "VRAM 부족. --util 을 낮추거나 --len 을 줄이십시오. 타 프로세스 점유도 확인."),
    (r"No available memory for the cache blocks|not enough KV cache|"
     r"Not enough KV cache memory",
     "가중치를 올리고 나니 KV 블록 자리가 없습니다. --util 을 올리거나 --len 을 줄이십시오."),
    (r"is larger than the maximum number of tokens|"
     r"model's max seq len .{0,40}larger than",
     "요청한 컨텍스트가 KV 풀보다 큽니다. --len 을 줄이거나 --util 을 올리십시오."),
    (r"Model architectures .{0,80}are not supported|"
     r"are not supported for now",
     "이 vLLM 이 모르는 아키텍처입니다. vLLM 을 올리거나 다른 모델을 쓰십시오."),
    (r"does not appear to have a file named config\.json|"
     r"Cannot find the config file",
     "config.json 이 없습니다. 디렉터리를 통째로 받았는지 확인하십시오."),
    # transformers 의 실제 문구를 그대로 쓴다 — 옵션 이름만 보면 설정 덤프에 걸린다.
    (r"set the option `?trust_remote_code=True`?|"
     r"requires you to execute custom code",
     "사용자 정의 코드가 필요합니다: --extra --trust-remote-code"),
    (r"CUDA compiler and CUDA toolkit headers are incompatible|"
     r"ninja: build stopped: subcommand failed",
     "JIT 커널 컴파일이 실패했습니다 (flashinfer 번들 헤더 ↔ CUDA 버전 불일치). "
     "VLLM_USE_FLASHINFER_SAMPLER=0 으로 우회합니다 — 이미 기본값입니다. "
     "다른 flashinfer 경로라면 VLLM_ATTENTION_BACKEND=FLASH_ATTN 도 시도하십시오."),
    (r"No such file or directory: '(ninja|nvcc|cc|g\+\+)'",
     "JIT 컴파일 도구를 찾지 못했습니다. .venv/bin 이 자식 PATH 에 없거나 "
     "ninja 가 없습니다: uv pip install --python .venv/bin/python ninja"),
    (r"max_num_seqs \((\d+)\) exceeds available Mamba cache blocks \((\d+)\)",
     "하이브리드 모델(Mamba/GDN)은 **디코드 시퀀스마다 Mamba 캐시 블록 1개**를 쓴다. "
     "util 을 낮추면 블록이 줄어드는데 --max-num-seqs 기본값(256)은 그대로라 충돌한다. "
     "로그가 알려준 블록 수보다 작게 --max-num-seqs 를 주십시오 (예: --max-num-seqs 192)."),
    (r"Engine core initialization failed|EngineCore failed to start|"
     r"EngineDeadError",
     "엔진 코어 초기화 실패. 로그 위쪽의 실제 예외를 보십시오."),
]

FATAL_SOFT = [
    (r"quantization|quantized",
     "양자화 처리 중 실패했습니다. 이 GPU 가 지원하는 스킴인지 확인하십시오 (docs/quantization_concepts.md)."),
    (r"tool.?call.?parser|tool_parser",
     "툴 콜 파서 문제입니다. --no-tools 로 끄거나 --tool-parser 를 바꿔보십시오."),
]

_HARD_RE = re.compile("|".join(f"(?:{p})" for p, _ in FATAL_HARD))
_SOFT_RE = re.compile("|".join(f"(?:{p})" for p, _ in FATAL_SOFT), re.IGNORECASE)


def scan_fatal(chunk):
    """새로 읽은 로그 조각에서 치명 오류를 찾는다. (발견 여부, 안내) 를 돌려준다.

    줄 단위로 본다. 여러 줄을 한 덩어리로 정규식에 넣으면 '.' 이 줄을 넘나들며
    서로 무관한 두 줄을 하나로 이어 붙여 매칭시키는 사고가 난다.
    """
    for line in chunk.splitlines():
        if _HARD_RE.search(line):
            for pat, hint in FATAL_HARD:
                if re.search(pat, line):
                    return True, hint
            return True, None
        if ERROR_MARK.search(line) and _SOFT_RE.search(line):
            for pat, hint in FATAL_SOFT:
                if re.search(pat, line, re.IGNORECASE):
                    return True, hint
            return True, None
    return False, None


RE_KV = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
RE_CONC = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
RE_INIT = re.compile(r"init engine \(profile, create kv cache, warmup model\) took\s*([\d.]+)\s*s")


def wait_ready(proc, log_path, host, port, timeout):
    """(준비됐나, 경과초, 로그전문). 로그를 흘리면서 /health 를 폴링한다."""
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/health"
    start = time.monotonic()
    offset = 0
    buf = []
    print(f"준비 대기 (최대 {timeout}s, /health 폴링 — 아래는 실시간 로그)")
    print("──────────────────────────────────────────────")
    ready = False
    fatal_hint = None
    while time.monotonic() - start < timeout:
        # 로그를 증분으로 읽어 그대로 흘린다. tail -f 프로세스를 띄우지 않으므로
        # 정리해야 할 자식도, trap 도 없다.
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
        except OSError:
            chunk = ""
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            buf.append(chunk)
            hit, hint = scan_fatal(chunk)
            if hit:
                fatal_hint = hint
                break
        if proc.poll() is not None:
            time.sleep(0.4)      # 마지막 로그가 다 써지도록
            break
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(1.0)
    # 남은 로그 마저 비우기
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            rest = f.read()
        if rest:
            sys.stdout.write(rest)
            buf.append(rest)
    except OSError:
        pass
    print("")
    print("──────────────────────────────────────────────")
    return ready, time.monotonic() - start, "".join(buf), fatal_hint


# ─────────────────────────────────────────────────────────────────────────────
# 9. 사후 보고 + bench 기록
# ─────────────────────────────────────────────────────────────────────────────
def report(log_text, args, util, quant, model, startup_s, gpu):
    section("KV 캐시 / 동시성")
    kv_tokens = conc = None
    m = RE_KV.search(log_text)
    if m:
        kv_tokens = int(m.group(1).replace(",", ""))
        print(f"  GPU KV 캐시  : {kv_tokens:,} 토큰")
    m = RE_CONC.search(log_text)
    if m:
        per_req = int(m.group(1).replace(",", ""))
        conc = float(m.group(2))
        print(f"  최대 동시성  : {per_req:,} 토큰/요청 기준 {conc:.2f}x")
    if kv_tokens:
        # llama.cpp 에서 -c 로 직접 정하던 값을, vLLM 에서는 여기서 역산해 본다.
        print("  → 컨텍스트별 동시 수용 시퀀스:")
        for ctx in (4096, 8192, 32768, 131072):
            if ctx <= kv_tokens:
                print(f"       {ctx:>7,} 토큰 → {kv_tokens // ctx:>4} 시퀀스")
    if not kv_tokens:
        print("  (로그에서 KV 캐시 크기를 찾지 못했습니다)")

    peak = None
    if gpu:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            peak = int(out.stdout.strip().splitlines()[0].strip())
        except Exception:
            pass

    # ── bench/runs.tsv (문서 §6 양식) ──
    # 손으로 적게 하면 안 적는다. 띄울 때마다 한 줄씩 자동으로 쌓아 두면
    # 손잡이 실측이 별도 노력 없이 굴러간다. ttft/tok_s/concurrency 는 부하를
    # 걸어야 나오는 값이라 여기서는 비워 둔다(-).
    os.makedirs(os.path.dirname(BENCH_TSV), exist_ok=True)
    header = ("date\tmodel\tquant\tutil\tmax_model_len\tenforce_eager\tconcurrency\t"
              "startup_s\tttft_ms\ttok_s\tpeak_vram_mib\tnote")
    new = not os.path.exists(BENCH_TSV)
    with open(BENCH_TSV, "a", encoding="utf-8") as f:
        if new:
            f.write(header + "\n")
        f.write("\t".join([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            os.path.basename(model.rstrip("/")),
            quant,
            f"{util:.2f}",
            str(args.max_model_len or "auto"),
            "1" if args.eager else "0",
            "-", f"{startup_s:.1f}", "-", "-",
            str(peak if peak is not None else "-"),
            f"kv={kv_tokens if kv_tokens else '-'}",
        ]) + "\n")
    print("")
    print(f"  기록        : {os.path.relpath(BENCH_TSV, SCRIPT_DIR)} 에 한 줄 추가")
    if peak is not None:
        print(f"  peak VRAM   : {peak} MiB")
    print(f"  기동 시간   : {startup_s:.1f}s"
          f"{'  (--enforce-eager 켜짐)' if args.eager else '  (CUDA 그래프 캡처 포함)'}")


def main(argv):
    args = parse_args(argv)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" vLLM 서버 기동")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    global _INTERACTIVE
    _INTERACTIVE = not args.yes
    if args.dry_run and args.port is None:
        args.port = 8000

    vllm_bin = find_vllm()

    # ★ 기존 vLLM 처리와 포트 선택을 GPU 측정보다 **먼저** 한다.
    # 옛 서버가 VRAM 을 쥔 채로 재면 util 이 터무니없이 낮게 잡히고,
    # 그 뒤에 죽여봐야 util 은 이미 고정된 뒤라 기동이 실패한다 (실측 사고).
    if not args.dry_run:
        handle_existing(True if args.force else args.kill_existing)
        section("포트")
        args.port = pick_port(args.port, args.host)
        print(f"  → {args.host}:{args.port}")

    gpu = gpu_info()
    section("GPU")
    if gpu:
        print(f"  {gpu[0]}  SM{gpu[1] * 10:.0f}  {gpu[2]} MiB"
              f"{f'  (타 프로세스 {gpu[3]} MiB)' if gpu[3] else ''}")
    else:
        print("  nvidia-smi 를 찾지 못했습니다. vLLM 은 GPU 없이 의미가 없습니다.")

    model = args.model or choose_model()
    config, weight_b, quant = inspect_model(model, gpu)
    util = decide_util(gpu, weight_b, args.util)
    parser = decide_tool_parser(model, config, args.tool_parser, args.no_tools)
    reasoning = decide_reasoning_parser(model, config, args.reasoning_parser,
                                        args.no_reasoning)

    section("KV 캐시")
    if any(a.startswith("--kv-cache-dtype") for a in args.extra):
        print("  → --extra 지정을 따름")
    elif args.kv_cache_dtype == "auto":
        print("  → auto (기본) — 모델 dtype 그대로. 손실 없음")
        # fp8 이 실제로 이득인 조건일 때만 권한다. 슬라이딩 윈도우가 많은 모델
        # (Gemma4 등)은 어차피 잘리는 KV 라 얻는 게 적고, SM80 Triton 백엔드는
        # fp8 KV 를 거부한다 — vLLM 자신이 그런 모델에는 권하지 않는다.
        if _fp8_kv_worth_it(config):
            print("     `--kv fp8` 이면 같은 util 에 토큰이 1.97배 들어갑니다")
            print("     (실측: 64k 문맥까지 PPL 0.987~0.999x · 정답과제 동일)")
    else:
        print(f"  → {args.kv_cache_dtype} — 토큰 밀도 1.97배")
        if not _fp8_kv_worth_it(config):
            print("     ⚠ 이 모델은 슬라이딩 윈도우 층이 많습니다. vLLM 은 이런 모델에")
            print("        fp8 KV 를 권하지 않고, SM80 Triton 백엔드는 거부할 수 있습니다.")
            print("        막히면 `--kv auto` 로 다시 실행하십시오.")

    cmd = build_cmd(vllm_bin, model, args, util, parser, reasoning)
    section("실행")
    print(format_cmd(cmd))
    if args.dry_run:
        print("")
        print("  --dry-run — 여기까지.")
        return 0

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(
        LOGS_DIR, f"vllm_{os.path.basename(model.rstrip('/'))}_{args.port}.log")
    print("")
    print(f"로그: {log_path}")
    env, extra_path, cuda_root = child_env()
    print(f"PATH 주입: {', '.join(os.path.relpath(x, SCRIPT_DIR) for x in extra_path)}"
          + (f"  ·  CUDA_HOME={os.path.relpath(cuda_root, SCRIPT_DIR)}" if cuda_root else ""))
    if env.get("VLLM_USE_FLASHINFER_SAMPLER") == "0":
        print("FlashInfer 샘플러: 끔 (JIT 컴파일 회피 — VLLM_USE_FLASHINFER_SAMPLER=1 로 켬)")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    print(f"기동 (PID {proc.pid})")
    print("")

    ready, elapsed, log_text, hint = wait_ready(
        proc, log_path, args.host, args.port, args.timeout)

    if ready:
        m = RE_INIT.search(log_text)
        if m:
            print(f"엔진 초기화: {float(m.group(1)):.1f}s (가중치 적재 + KV 생성 + 워밍업)")
        report(log_text, args, util, quant, model, elapsed, gpu)
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f" 준비 완료  http://{args.host}:{args.port}/v1   (PID {proc.pid})")
        served = args.name or os.path.basename(os.path.abspath(model.rstrip("/")))
        print(f"   모델 : {served}")
        print(f"   확인 : curl -s http://{args.host}:{args.port}/v1/models")
        print(f"   종료 : kill {proc.pid}")
        print(f"   로그 : tail -f {log_path}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return 0

    if proc.poll() is not None:
        print(f"서버가 기동 중 종료했습니다 (exit {proc.returncode}).", file=sys.stderr)
        if hint:
            print(f"  → {hint}", file=sys.stderr)
        print(f"  로그: {log_path}", file=sys.stderr)
        return 1

    if hint:
        # 치명 패턴은 봤지만 프로세스는 아직 살아 있다. 정리하고 원인을 알려준다.
        print("치명적 오류를 감지했습니다.", file=sys.stderr)
        print(f"  → {hint}", file=sys.stderr)
        try:
            proc.terminate()
        except OSError:
            pass
        print(f"  로그: {log_path}", file=sys.stderr)
        return 1

    # 타임아웃. 죽이지 않는다 — 큰 모델은 정말 더 걸릴 수 있다.
    print(f"{args.timeout}s 안에 준비되지 않았습니다. 서버는 계속 로딩 중입니다 "
          f"(PID {proc.pid}).")
    print(f"  상태 : curl -s -o /dev/null -w '%{{http_code}}' "
          f"http://{args.host}:{args.port}/health   (준비되면 200)")
    print(f"  로그 : tail -f {log_path}")
    print(f"  종료 : kill {proc.pid}")
    print(f"  더 기다리려면: --timeout {args.timeout * 2}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n중단했습니다.", file=sys.stderr)
        sys.exit(130)
