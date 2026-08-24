#!/usr/bin/env python3
"""models/<이름>/ 아래로 모델을 받는다. 저장소 URL 하나만 주면 갈래는 스스로 정한다.

  safetensors 저장소 → vLLM 용. **저장소 전체**를 받는다 (샤드 + config + 토크나이저).
                       받기 전에 config.json 만 먼저 읽어 MLX·양자화·크기를 판정한다.
  GGUF 저장소        → llama.cpp 용. 양자화 수준을 골라 파일 1~3개(본체/mmproj/MTP)를 받는다.

두 갈래가 다운로드·이어받기·재시도·sha256 검증 인프라를 공유한다. 갈래마다 따로
구현하면 규칙이 어긋난다 — 아래에 적힌 것과 같은 일이 실제로 벌어졌다.

download_model.sh 의 파이썬 판이다. 두 파일은 같은 규칙을 따라야 한다:
동작이 갈리면 sh 가 기준이다.

예전 파이썬 판이 sh 와 달랐던 점(전부 여기서 맞췄다):
  - 재시도가 죽어 있었다. except 절 끝의 `break` 가 첫 실패에서 루프를 빠져나가
    "3회 재시도" 가 한 번도 돌지 않았다. sh 의 [FIX 2] 와 같은 종류의 버그다.
  - 실패하면 받다 만 파일을 지웠다. 그러면 다음 실행에서 이어받을 수가 없다.
  - Range 요청을 보내고 서버 응답 코드를 안 봤다. 서버가 206 대신 200(전체)을
    돌려주면 기존 파일 뒤에 처음부터 다시 붙여서 파일을 조용히 망가뜨린다.
    → 이어받기는 wget -c / curl -C - 에게 맡긴다. 이미 검증된 구현이다.
  - HF_TOKEN(gated 저장소), sha256 검증, 모델/프로젝션 동시 다운로드가 없었다.
"""
import concurrent.futures
import getpass
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SKIP_VERIFY = os.environ.get("SKIP_VERIFY", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# MTP(다중 토큰 예측) 사이드카 판별
#
# 예전 정규식은 (^|[-_.])mtp([-_.]|$) 였다. 구분자로 둘러싸인 'mtp' 만 인정하므로
# HauhauCS 저장소의 'Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf'
# 를 놓쳤다. 그 파일은 MTP 선택 목록에 아예 뜨지 않았고(=받을 방법이 없었고),
# 대신 본 모델 목록에 양자화 'unknown' 으로 섞여 들어갔다.
#
# 이제 구분자로 시작하는 토큰이 'mtp'(뒤에 숫자 허용)로 끝나면 사이드카로 본다.
# 'MTP', 'FastMTP', 'mtp2' 가 모두 걸리고, 'Q6_K_P' 같은 양자화 태그는 안 걸린다.
# ─────────────────────────────────────────────────────────────────────────────
MTP_RE = re.compile(r"(?:^|[-_.])[A-Za-z]*mtp[0-9]*(?:[-_.]|$)", re.IGNORECASE)


def is_mtp_name(name):
    """파일 이름(또는 저장소 이름)이 MTP 를 가리키면 True."""
    return bool(MTP_RE.search(os.path.basename(name)))


# ─────────────────────────────────────────────────────────────────────────────
# HF 토큰
#
# gated 저장소(Llama, Gemma, FLUX 계열 등)는 토큰이 있어야 받아진다. 예전에는
# 토큰을 환경변수로만 받았고, 게다가 다운로드 요청에만 붙였다. 그래서 sha256 정답지를
# 가져오는 HF API 호출이 401 을 받고 빈 문자열을 돌려줬고, 화면에는
# "sha256 정보 없음 — 검증 생략" 만 떴다. 정작 검증이 가장 필요한 gated 20GB 짜리에서
# 검증이 조용히 사라지는 셈이다. 이제 API 호출에도 같은 토큰을 붙인다.
#
# 출처 우선순위: HF_TOKEN → HUGGING_FACE_HUB_TOKEN → huggingface-cli 로그인 파일
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_hf_token():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = os.path.join(os.environ.get("HF_HOME") or
                        os.path.expanduser("~/.cache/huggingface"), "token")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


HF_TOKEN = _resolve_hf_token()


def _auth_header():
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def prompt(msg):
    try:
        return input(msg).strip()
    except EOFError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 다운로드 — 이어받기는 wget -c / curl -C - 가 한다.
# wget -c 는 Content-Range 를 확인하고 이어받으며, HF 의 302 → 서명 CDN 리다이렉트
# 너머로도 정상 동작하는 것을 실측 확인했다(206 Partial Content + sha256 일치).
#
# 진행 표시는 우리가 직접 그린다. 예전에는 wget/curl 이 각자 자기 진행률을 같은
# 터미널에 그렸는데, 이 스크립트는 모델·mmproj·MTP 를 **동시에** 받으므로 세 프로세스가
# 서로 커서를 뺏어 한 줄에 겹쳐 찍혔다. 그래서 받는 쪽은 조용히(-q) 돌리고
# 화면은 여기서 혼자 그린다 — 파일 하나당 한 줄, 매번 같은 자리를 다시 쓴다.
#
# 진행률은 다운로더 출력을 파싱하지 않고 목적지 파일 크기를 직접 잰다. 전체 크기는
# HF 트리 API 가 준 값(size, 실제 파일 크기와 바이트까지 일치하는 것을 실측 확인)이라
# 정확하고, 이어받기(부분 파일)도 그냥 맞아떨어진다. 파싱할 게 없으니 wget/curl 의
# 버전·로케일 차이도 안 탄다.
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _fmt_eta(sec):
    if not sec or sec <= 0 or sec > 99 * 3600:
        return "--:--"
    minutes, seconds = divmod(int(sec), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class Job:
    """받을 파일 하나. 상태는 다운로드 스레드가 쓰고 그리는 쪽이 읽는다.

    파이썬 문자열/정수 대입은 원자적이라 잠금이 필요 없다(읽는 쪽은 한 프레임
    늦은 값을 봐도 아무 문제가 없다).
    """

    def __init__(self, label, url, dest, total=0, sha="", resumed=0):
        self.label = label
        self.url = url
        self.dest = dest
        self.total = total or 0
        self.sha = sha
        self.resumed = resumed        # 시작 시점에 이미 있던 바이트 수
        self.state = "대기"
        self.note = ""
        self.ok = False
        self.err = ""
        self.speed = 0.0
        self._last = None             # (시각, 바이트)

    def current(self):
        try:
            return os.path.getsize(self.dest)
        except OSError:
            return 0

    def tick(self, now):
        """0.3초 이상 지났으면 속도를 갱신한다(지수 평활)."""
        cur = self.current()
        if self._last is None:
            self._last = (now, cur)
            return cur
        dt = now - self._last[0]
        if dt >= 0.3:
            inst = max(0, cur - self._last[1]) / dt
            self.speed = inst if self.speed == 0 else self.speed * 0.7 + inst * 0.3
            self._last = (now, cur)
        return cur


def _downloader_cmd(url, dest):
    """(cmd, 도구이름) 또는 (None, None). 진행 표시는 우리가 그리므로 조용히 돌린다."""
    if shutil.which("wget"):
        cmd = ["wget", "-q"]
        if HF_TOKEN:
            cmd.append(f"--header=Authorization: Bearer {HF_TOKEN}")
        return cmd + ["-c", "-O", dest, url], "wget"
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "-sS"]
        if HF_TOKEN:
            cmd += ["-H", f"Authorization: Bearer {HF_TOKEN}"]
        if os.path.exists(dest):
            cmd += ["-C", "-"]
        return cmd + ["-o", dest, url], "curl"
    return None, None


def download(job, max_retries=3):
    """한 파일을 받는다. 화면에 직접 찍지 않고 job 의 상태만 바꾼다."""
    for attempt in range(1, max_retries + 1):
        cmd, _tool = _downloader_cmd(job.url, job.dest)
        if cmd is None:
            job.state, job.note = "실패", "curl 또는 wget 이 필요합니다"
            return False
        job.state = "받는 중"
        if attempt > 1:
            job.note = f"재시도 {attempt}/{max_retries}"
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            job.state, job.note, job.ok = "완료", "", True
            return True
        job.err = (proc.stderr or b"").decode("utf-8", "replace").strip()

        # 인증 문제라면 세 번 더 해봐야 세 번 다 실패한다. 즉시 원인을 알려주고 빠져나온다.
        # (여기서 화면에 찍으면 진행 표시가 깨지므로 note 에만 남긴다.)
        if not HF_TOKEN and _is_gated(job.url):
            job.state, job.note = "실패", "gated 저장소 — HF_TOKEN 이 필요합니다"
            return False

        if attempt < max_retries:
            job.state = "대기"
            for remain in range(attempt * 10, 0, -1):
                job.note = f"실패 — {remain}초 후 재시도 ({attempt + 1}/{max_retries})"
                time.sleep(1)

    job.state, job.note = "실패", "최대 재시도 횟수 초과"
    # 여기서 파일을 지우지 않는다. 부분 파일이 남아 있어야 다음 실행에서 이어받을 수 있고,
    # sha256 검증이 "받다 만 것"인지 "깨진 것"인지 구분해 준다.
    return False


def _render(jobs, width):
    # 라벨 폭은 고정 7 이 아니라 실제 라벨에 맞춘다. GGUF 갈래는 'model'/'mmproj'/'mtp'
    # 뿐이라 7 로 충분했지만, safetensors 갈래는 파일명을 그대로 라벨로 쓴다.
    lw = max(7, max(len(j.label) for j in jobs))
    lines = []
    for j in jobs:
        head = f"{j.label:<{lw}}"
        if j.state in ("완료", "실패"):
            mark = "✓" if j.ok else "✗"
            lines.append(f"{head} {mark} {j.state}{('  ' + j.note) if j.note else ''}")
            continue
        cur = j.current()
        if j.total:
            frac = min(1.0, cur / j.total)
            filled = int(24 * frac)
            bar = "█" * filled + "░" * (24 - filled)
            eta = (j.total - cur) / j.speed if j.speed > 0 else 0
            line = (f"{head} {bar} {frac * 100:5.1f}%  "
                    f"{_fmt_bytes(cur)}/{_fmt_bytes(j.total)}  "
                    f"{_fmt_bytes(j.speed)}/s  ETA {_fmt_eta(eta)}")
        else:
            line = f"{head} {_fmt_bytes(cur)}  {_fmt_bytes(j.speed)}/s"
        lines.append(line + (f"  {j.note}" if j.note else ""))
    return [ln[:width] for ln in lines]


def run_jobs(jobs, max_parallel=None):
    """모든 잡을 동시에 받으면서 잡마다 한 줄씩 진행률을 그린다.

    max_parallel 을 주면 그 수만큼만 동시에 받는다. GGUF 갈래는 파일이 1~3개라
    전부 동시에 받아도 됐지만, safetensors 저장소는 15개가 넘을 수 있다. 대역폭은
    그대로인데 연결만 15개로 쪼개면 개별 파일이 전부 느려져서, 하나가 끝나 검증을
    시작할 수 있는 시점만 늦어진다. 기본값(None)은 예전 동작 그대로다.
    """
    if not jobs:
        return True
    tty = sys.stdout.isatty()
    drawn = 0
    last_plain = 0.0

    workers = min(len(jobs), max_parallel) if max_parallel else len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, j) for j in jobs]
        while True:
            running = not all(f.done() for f in futures)
            now = time.monotonic()
            for j in jobs:
                j.tick(now)
            if tty:
                # 창 크기는 매 프레임 다시 본다. 줄이 터미널 폭을 넘으면 자동 줄바꿈이
                # 일어나 '올릴 줄 수'가 어긋나므로, 폭보다 한 칸 짧게 잘라서 그린다.
                width = max(40, shutil.get_terminal_size((100, 24)).columns - 1)
                # 이미 그린 줄 수만큼 커서를 올리고(ESC[nA) 각 줄을 지운 뒤 다시 쓴다.
                out = [f"\033[{drawn}A"] if drawn else []
                lines = _render(jobs, width)
                out += ["\r\033[2K" + ln + "\n" for ln in lines]
                sys.stdout.write("".join(out))
                sys.stdout.flush()
                drawn = len(lines)
            elif now - last_plain >= 20 or not running:
                # 로그로 리다이렉트된 경우. 커서 제어는 쓰레기만 남기므로 주기적으로 한 줄씩.
                for ln in _render(jobs, 200):
                    print(ln, flush=True)
                last_plain = now
            if not running:
                break
            time.sleep(0.4)

    ok = all(j.ok for j in jobs)
    for j in jobs:
        if not j.ok and j.err:
            print("", file=sys.stderr)
            print(f"[{j.label}] {j.url}", file=sys.stderr)
            for ln in j.err.splitlines()[-5:]:
                print(f"  {ln}", file=sys.stderr)
    return ok


def _is_gated(url):
    """gated 저장소 판별 (요청 1회). 조용히 참/거짓만 돌려준다."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "wget/1.21",
                                              **_auth_header()})
        urllib.request.urlopen(req, timeout=15)
        return False
    except urllib.error.HTTPError as e:
        return e.code in (401, 403)
    except Exception:
        return False


def _diagnose_gated(url):
    if not _is_gated(url):
        return False
    print("", file=sys.stderr)
    print("⛔ 이 저장소는 로그인/약관 동의가 필요한 gated 저장소입니다.", file=sys.stderr)
    print("   HF 웹에서 해당 모델의 약관에 동의한 뒤 토큰을 주고 재실행하세요:", file=sys.stderr)
    print("     HF_TOKEN=hf_xxx python3 download_model.py", file=sys.stderr)
    return True


def ensure_token(model_url):
    """gated 여부는 20GB 를 태우고 나서가 아니라 시작 전에 알아야 한다.

    다운로드는 스레드로 도는데 거기서 토큰을 물어볼 수는 없다(입력이 섞인다).
    그래서 아직 프롬프트가 안전한 이 지점에서 한 번 확인한다. 요청 1회.
    """
    global HF_TOKEN
    if HF_TOKEN or not _is_gated(model_url):
        return
    print("")
    print("⛔ 이 저장소는 로그인/약관 동의가 필요한 gated 저장소입니다.")
    print("   먼저 HF 웹에서 해당 모델의 약관에 동의했는지 확인하세요.")
    print("   토큰: https://huggingface.co/settings/tokens (read 권한이면 충분)")
    tok = getpass.getpass("HF 토큰을 붙여넣으세요 (그냥 Enter 면 중단): ").strip()
    if not tok:
        print("토큰 없이는 이 저장소를 받을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    HF_TOKEN = tok


# ─────────────────────────────────────────────────────────────────────────────
# 무결성 검증
#
# HuggingFace API 의 lfs.oid 는 파일 내용의 sha256 이다. 즉 정답지를 서버가 준다.
# 이걸 안 쓰면 20GB 짜리 GGUF 가 조용히 잘려도 그 자리에서는 알 수 없고,
# 나중에 llama-server 가 'invalid magic' 이나 알 수 없는 로드 실패로만 알려준다.
# ─────────────────────────────────────────────────────────────────────────────
def _sha256(path):
    # openssl 이 파이썬 hashlib/coreutils sha256sum 보다 빠르다. 같은 12.7GB 파일 실측:
    #     sha256sum  78.6s (154 MB/s)   ← 이식성 위주의 C 구현
    #     openssl    29.0s (419 MB/s)   ← 어셈블리(AVX2) 최적화
    # 이 CPU(Coffee Lake)에는 SHA-NI 확장이 없어서 구현 차이가 그대로 드러난다.
    if shutil.which("openssl"):
        print(f"  검증 중 (sha256 via openssl, 약 2.4초/GB): {os.path.basename(path)}", flush=True)
        out = subprocess.run(["openssl", "dgst", "-sha256", path],
                             capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip().split()[-1]
    print(f"  검증 중 (sha256, 약 6.5초/GB): {os.path.basename(path)}", flush=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha(path, want):
    """일치하거나 검증 불가면 True, 불일치면 False."""
    if not os.path.isfile(path):
        return False
    if not want:
        print("  (sha256 정보 없음 — 검증 생략)")
        return True
    if SKIP_VERIFY:
        print("  (SKIP_VERIFY=1 — 검증 생략)")
        return True
    # HF 는 gated 저장소의 해시를 별표로 가린다. 그대로 비교하면 항상 불일치가 된다.
    if not re.fullmatch(r"[0-9a-fA-F]{64}", want):
        print("  (HF 가 해시를 가린 저장소 — 검증 생략)")
        return True
    got = _sha256(path)
    if got == want:
        print(f"  ✓ sha256 일치: {os.path.basename(path)}")
        return True
    print(f"  ✗ sha256 불일치: {os.path.basename(path)}", file=sys.stderr)
    print(f"     기대: {want}", file=sys.stderr)
    print(f"     실제: {got}", file=sys.stderr)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 이름 짓기
#
# 예전에는 디렉터리 이름을 사람이 먼저 타이핑하게 했다. 그 결과 models/ 안에는
# 양자화 수준이 빠진 디렉터리(Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)와
# 제작자가 빠진 디렉터리가 섞여 남았다. 사람이 매번 정확히 옮겨 적을 이유가 없다 —
# URL 이 그 정보를 이미 다 갖고 있다.
#
# 규칙: <제작자>_<파일명 stem>  (stem 안에 제작자가 이미 있으면 중복은 지운다)
#   https://huggingface.co/HauhauCS/Qwen3.8-...-MTP-GGUF/resolve/main/
#     Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf
#   →  HauhauCS_Qwen3.8-27B-Uncensored-Aggressive-Q6_K_P
# ─────────────────────────────────────────────────────────────────────────────
def url_basename(url):
    """URL → 파일 이름 (쿼리스트링 ?download=true 제거, %XX 디코드)."""
    path = urllib.parse.urlparse(url).path
    return urllib.parse.unquote(os.path.basename(path))


def normalize_hf_url(url):
    """Hugging Face blob 웹페이지 URL을 실제 resolve 다운로드 URL로 변환한다."""
    parsed = urllib.parse.urlsplit(url)
    parts = parsed.path.split("/")
    if parsed.netloc == "huggingface.co" and "blob" in parts:
        parts[parts.index("blob")] = "resolve"
        parsed = parsed._replace(path="/".join(parts))
        return urllib.parse.urlunsplit(parsed)
    return url


def hf_owner_from_url(url):
    """https://huggingface.co/<owner>/<repo>/... → owner."""
    p = urllib.parse.urlparse(url)
    if p.netloc != "huggingface.co":
        return ""
    parts = [x for x in p.path.split("/") if x]
    if parts and parts[0] in ("datasets", "spaces", "models"):
        parts = parts[1:]
    return parts[0] if len(parts) >= 2 else ""


def hf_repo_from_url(url):
    """https://huggingface.co/<owner>/<repo>/... → repo name."""
    p = urllib.parse.urlparse(url)
    if p.netloc != "huggingface.co":
        return ""
    parts = [x for x in p.path.split("/") if x]
    if parts and parts[0] in ("datasets", "spaces", "models"):
        parts = parts[1:]
    return parts[1] if len(parts) >= 2 else ""


def sanitize_name(name):
    """파일 이름으로 쓸 수 없는 문자를 _ 로. 앞뒤 구분자도 정리한다."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("-._")


def _strip_token(stem, token):
    """stem 에서 토큰 하나를 구분자 경계로 제거 (대소문자 무시).

    'HauhauCS' 를 지울 때 'HauhauCSX' 같은 다른 단어를 건드리면 안 되므로 경계를 본다.
    """
    if not token:
        return stem
    pat = re.compile(r"(^|[-_.])" + re.escape(token) + r"([-_.]|$)", re.IGNORECASE)
    return pat.sub(r"\1", stem, count=1).strip("-._")


def suggest_name(model_url):
    """모델 URL → 제안할 디렉터리/파일 이름."""
    stem = re.sub(r"\.gguf$", "", url_basename(model_url), flags=re.IGNORECASE)
    if not stem:
        return ""
    owner = hf_owner_from_url(model_url)
    if owner:
        stem = f"{owner}_{_strip_token(stem, owner)}"
    repo = hf_repo_from_url(model_url)
    if is_mtp_name(repo) and not is_mtp_name(stem):
        stem = f"{stem}-MTP"
    return sanitize_name(stem)


def proj_tag(basename):
    """프로젝션 파일명에서 정밀도 태그만 뽑는다 (f16, bf16, q8_0 ...).

    보통 이름 끝쪽에 붙으므로 마지막 매치를 쓴다.
    """
    low = re.sub(r"\.gguf$", "", basename, flags=re.IGNORECASE).lower()
    found = re.findall(r"bf16|fp16|f16|fp32|f32|q[0-9]+(?:_[0-9a-z]+)*", low)
    # 하이픈은 반드시 남긴다. run_llama_server.sh 는 '*mmproj-*.gguf' 로 찾는다.
    return f"mmproj-{found[-1] if found else 'default'}"


# ─────────────────────────────────────────────────────────────────────────────
# 대화형 입력 — URL 을 먼저 받고, 이름은 거기서 뽑아 제안한다.
# ─────────────────────────────────────────────────────────────────────────────
def ask_names(model_url, proj_url, name_suffix=""):
    if not url_basename(model_url).lower().endswith(".gguf"):
        print("⚠ 모델 URL 이 .gguf 로 끝나지 않습니다. 이름이 이상하게 잡힐 수 있습니다.")
    suggested = suggest_name(model_url)
    if suggested and name_suffix and not is_mtp_name(suggested):
        suggested = f"{suggested}{name_suffix}"
    if suggested:
        print("")
        print("URL 에서 뽑은 이름:")
        print(f"  디렉터리  : models/{suggested}/")
        print(f"  모델 파일 : {suggested}.gguf")
        if proj_url:
            print(f"  프로젝션  : {suggested}_{proj_tag(url_basename(proj_url))}.gguf")
        ans = prompt("이 이름을 쓸까요? (Y/n, 또는 원하는 이름을 직접 입력): ")
        low = ans.lower()
        if ans == "" or low in ("y", "yes"):
            return suggested
        if low not in ("n", "no"):
            return sanitize_name(ans.rstrip("/"))

    name = sanitize_name(prompt("Model directory name (will create models/<name>/): ").rstrip("/"))
    if not name:
        print("Model directory name is required.", file=sys.stderr)
        sys.exit(1)
    return name


# ─────────────────────────────────────────────────────────────────────────────
# 이미 있는 파일 처리 — 건너뛰기 / 이어받기
#
# 예전에는 목적지에 파일이 있으면 무조건 "Overwrite? (y/N)" 를 물었고,
#   y → os.remove() 로 지우고 처음부터   N → 본 모델이면 통째로 중단
# 이었다. 그래서 30GB 를 25GB 까지 받다 끊긴 뒤 다시 실행하면, 받아 둔 25GB 를
# 버리고 처음부터 받거나 아예 아무것도 못 하거나 둘 중 하나였다.
# 실패 메시지의 "다시 실행하면 이어받습니다" 는 사실이 아니었던 셈이다.
# (이어받기 자체는 wget -c 가 잘 한다. 실측: 부분 파일 → 206 Partial Content,
#  이어받은 파일의 sha256 이 새로 받은 것과 일치.)
#
# 이제 HF 가 알려준 기대 크기와 실제 파일 크기를 비교해서 스스로 정한다:
#   같다      → 건너뛴다 (VERIFY_ALL=1 이면 sha256 도 확인, REDOWNLOAD=1 이면 새로 받는다)
#   작다      → 이어받는다 (묻지 않는다)
#   크다/모름 → 그때만 사람에게 묻는다
# ─────────────────────────────────────────────────────────────────────────────
REDOWNLOAD = os.environ.get("REDOWNLOAD", "0") == "1"
VERIFY_ALL = os.environ.get("VERIFY_ALL", "0") == "1"


def plan_file(label, url, dest, total, sha):
    """받을 Job 을 돌려준다. 받을 필요가 없으면 None."""
    name = os.path.basename(dest)
    if not os.path.exists(dest):
        return Job(label, url, dest, total, sha)

    cur = os.path.getsize(dest)
    if total and cur == total and not REDOWNLOAD:
        print(f"  {label:<7} 이미 받아져 있습니다 ({_fmt_bytes(cur)}) — 건너뜁니다: {name}")
        return None
    if total and cur < total:
        pct = cur / total * 100
        print(f"  {label:<7} 받다 만 파일 {_fmt_bytes(cur)}/{_fmt_bytes(total)} "
              f"({pct:.1f}%) — 이어받습니다: {name}")
        return Job(label, url, dest, total, sha, resumed=cur)
    if total and cur == total and REDOWNLOAD:
        print(f"  {label:<7} REDOWNLOAD=1 — 지우고 새로 받습니다: {name}")
        os.remove(dest)
        return Job(label, url, dest, total, sha)

    # 기대 크기보다 크거나, 크기를 아예 모르는 경우(비 HF URL 등)만 사람에게 묻는다.
    why = (f"기대({_fmt_bytes(total)})보다 큽니다" if total else "기대 크기를 알 수 없습니다")
    print(f"  {label:<7} {name}: {_fmt_bytes(cur)} — {why}.")
    if not (prompt("          지우고 새로 받을까요? (y/N): ") or "N").lower().startswith("y"):
        print(f"  {label:<7} 그대로 두고 건너뜁니다.")
        return None
    os.remove(dest)
    return Job(label, url, dest, total, sha)


def hf_repo_from_input(url):
    """Return (owner, repo, revision) from an HF repo/tree URL."""
    parsed = urllib.parse.urlparse(url.rstrip("/"))
    if parsed.netloc != "huggingface.co":
        return "", "", "main"
    parts = [x for x in parsed.path.split("/") if x]
    if parts and parts[0] in ("models", "datasets", "spaces"):
        parts = parts[1:]
    if len(parts) < 2:
        return "", "", "main"
    revision = "main"
    if len(parts) >= 4 and parts[2] in ("tree", "blob", "resolve"):
        revision = parts[3]
    return parts[0], parts[1], revision


def list_repo_tree(repo_url):
    """저장소의 **모든** 파일 목록을 HF 공개 트리 API 로 한 번에 가져온다.

    예전에는 이 함수가 .gguf 만 골라서 돌려줬다(list_repo_gguf_files). 그런데
    safetensors 저장소는 "가중치 샤드 + config.json + tokenizer* + 채팅 템플릿"을
    **통째로** 받아야 하므로(§3.1), 확장자로 미리 거르면 갈래를 나눌 수조차 없다.
    그래서 여기서는 거르지 않고 전부 돌려주고, 무엇을 받을지는 갈래별로 정한다.

    돌려주는 meta 는 {경로: (크기, sha256)}. HF 는 LFS 파일에만 lfs.oid(=sha256)를
    주므로 작은 설정 파일(config.json 등)은 sha 가 빈 문자열이다 — 검증은 큰 파일에만
    걸리는데, 조용히 잘려서 문제가 되는 것도 큰 파일뿐이라 실질 손해가 없다.
    """
    owner, repo, revision = hf_repo_from_input(repo_url)
    if not owner or not repo:
        print("Hugging Face repository URL is invalid.", file=sys.stderr)
        sys.exit(1)
    api = f"https://huggingface.co/api/models/{owner}/{repo}/tree/{revision}?recursive=true"
    req = urllib.request.Request(api, headers=_auth_header())
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            entries = json.load(response)
    except Exception as exc:
        print(f"Could not read repository file list: {exc}", file=sys.stderr)
        sys.exit(1)
    files = sorted(e["path"] for e in entries if e.get("type") == "file")
    # 크기와 sha256(lfs.oid)을 여기서 같이 챙긴다. 이 한 번의 응답에 다 들어 있으므로
    # 파일마다 API 를 또 부를 이유가 없고, 크기를 알면 '이미 다 받았는지 / 어디까지
    # 받다 말았는지' 를 사람에게 묻지 않고 판단할 수 있다.
    meta = {e["path"]: (e.get("size") or 0, (e.get("lfs") or {}).get("oid", "") or "")
            for e in entries if e.get("type") == "file"}
    return owner, repo, revision, files, meta


def hf_file_url(owner, repo, revision, path):
    return (f"https://huggingface.co/{owner}/{repo}/resolve/"
            f"{urllib.parse.quote(revision)}/{urllib.parse.quote(path, safe='/')}?download=true")


def quantization_label(path):
    stem = os.path.basename(path).rsplit(".", 1)[0]
    found = re.findall(r"(?:^|[-_.])(Q[0-9]+(?:_[0-9A-Za-z]+)*|IQ[0-9]+(?:_[0-9A-Za-z]+)*|BF16|F16|F32)(?:[-_.]|$)", stem, re.IGNORECASE)
    return found[-1].upper() if found else "unknown"


def choose_file(label, files):
    print(f"\n{label} 선택:")
    for index, path in enumerate(files, 1):
        print(f"  [{index}] {os.path.basename(path)}")
    while True:
        answer = prompt(f"번호 선택 [기본값 1]: ") or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(files):
            return files[int(answer) - 1]
        print("올바른 번호를 선택하세요.")


# ═════════════════════════════════════════════════════════════════════════════
# safetensors 저장소 갈래 (vLLM 용)
#
# GGUF 갈래와 근본적으로 다른 점이 넷이다:
#
#   ① 받는 단위가 '파일'이 아니라 '저장소 전체'다.
#      GGUF 는 한 파일에 가중치·토크나이저·채팅템플릿이 다 들어있다. safetensors 는
#      텐서만 들고 나머지는 옆의 JSON 들이 나눠 갖는다. 그래서 고를 게 없고,
#      대신 무엇을 빼야 하는지(§파일 필터)가 중요해진다.
#
#   ② 양자화를 고를 수 없다.
#      GGUF 저장소는 한 곳에 Q4_K_M / Q6_K / Q8_0 가 나란히 있어서 파일을 고르지만,
#      HF 는 AWQ / GPTQ 가 아예 **다른 저장소**다. 저장소를 고른 시점에 양자화가
#      이미 정해져 있다. 그래서 choose_file 같은 UI 가 설 자리가 없다.
#
#   ③ 프로젝션 모델(mmproj)을 따로 받지 않는다.
#      비전 타워와 프로젝터가 같은 샤드 안에 들어 있다. preprocessor_config.json 이
#      있으면 그 저장소는 VLM 이고, vLLM 은 아무것도 추가로 지정하지 않아도 올린다.
#
#   ④ 실패의 대가가 다르다. ← 이게 설계를 지배한다
#      GGUF 는 커도 llama.cpp 의 -ngl 로 일부만 GPU 에 올릴 수 있으니 "일단 받아보고
#      맞춘다"가 가능했다. vLLM 에는 그 손잡이가 사실상 없다(--cpu-offload-gb 는
#      존재하지만 매 forward 마다 가중치를 PCIe 로 실어 나르는 방식이라 이 박스의
#      Gen2 x4 에서는 쓸 물건이 아니다). 안 들어가면 받은 30GB 가 그냥 낭비다.
#      → 그래서 가중치를 1바이트도 받기 전에 config.json(1KB)만 먼저 읽고 판정한다.
# ═════════════════════════════════════════════════════════════════════════════

WEIGHT_EXTS = (".safetensors", ".bin", ".pth", ".pt")

# 받지 않을 것. 안 거르면 다운로드가 2배가 되는데, 어느 쪽도 vLLM 이 쓰지 않는다.
EXCLUDE_SUFFIX = (
    ".gguf",            # llama.cpp 용
    ".onnx", ".onnx_data",
    ".msgpack",         # Flax
    ".h5", ".ckpt",     # TensorFlow / 구 체크포인트
    ".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf",
    ".gitattributes",
)
# 원본 체크포인트 디렉터리. safetensors 와 같은 가중치를 한 벌 더 들고 있다.
EXCLUDE_DIR_PREFIX = ("original/", "originals/")


def _fetch_json(url, timeout=20):
    """작은 JSON 하나를 받아 dict 로. 실패하면 None (판정 불가로 처리)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wget/1.21",
                                                   **_auth_header()})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def detect_repo_kind(files):
    """저장소가 어느 갈래인지. 사용자가 미리 알 필요가 없게 스스로 판정한다."""
    low = [f.lower() for f in files]
    has_st = any(f.endswith(".safetensors") for f in low)
    has_gguf = any(f.endswith(".gguf") for f in low)
    if has_st and has_gguf:
        return "both"
    if has_st:
        return "safetensors"
    if has_gguf:
        return "gguf"
    # safetensors 이전 형식만 올라온 저장소도 있다. vLLM 은 이것도 읽는다.
    if any(f.endswith((".bin", ".pth")) for f in low):
        return "safetensors"
    return "unknown"


def _gpu_total_mib():
    """이 머신의 VRAM(MiB). nvidia-smi 가 없으면 0 — 판정을 건너뛴다.

    다운로드는 GPU 가 없는 곳에서 돌릴 수도 있으므로 실패를 오류로 만들지 않는다.
    """
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return 0


def _gpu_compute_cap():
    """이 GPU 의 compute capability(예: 8.0). 알 수 없으면 0.0.

    fp8 판정에 필요하다. fp8 은 '지원되냐'가 아니라 '텐서코어가 있냐'의 문제라서
    카드 세대를 봐야 정확한 답이 나온다 — SM80(A100/GA100)에는 fp8 연산기가 없어서
    fp8 가중치를 써도 이득이 0 이지만, SM89(Ada)/SM90(Hopper) 부터는 이득이 크다.
    """
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return 0.0


def _float_quant_bits(method, qc, repo):
    """부동소수점 양자화면 그 비트 폭(4 또는 8), 아니면 None.

    quant_method 만 봐서는 알 수 없다. 'compressed-tensors' 도 'modelopt' 도 포맷
    이름일 뿐이고 그 안에 int4 도 int8 도 fp8 도 fp4 도 들어갈 수 있다. 실측한 두 사례:

      RedHatAI/…-FP8          quant_method='compressed-tensors', weights={type:float, num_bits:8}
      Blackfrost-AI/…-NVFP4   quant_method='modelopt',           weights={type:float, num_bits:4}

    포맷 이름만 보면 앞의 것은 '◎ 1순위', 뒤의 것은 '△ 대개 fp8 계열' 이라고 찍힌다.
    둘 다 틀렸다 — 이 카드(SM80)에서 앞은 이득 0 이고 뒤는 아예 못 돈다. 그래서
    config_groups 안의 실제 타입과 비트 폭을 본다.
    """
    for group in (qc.get("config_groups") or {}).values():
        w = (group or {}).get("weights") or {}
        if str(w.get("type", "")).lower() == "float":
            bits = w.get("num_bits")
            if bits in (4, 8):
                return bits
    # config_groups 가 없는 형식은 이름으로 판단한다.
    if "fp8" in method or re.search(r"(^|[-_.])fp8", repo, re.IGNORECASE):
        return 8
    if "fp4" in method or re.search(r"(^|[-_.])(nv)?fp4|mxfp4", repo, re.IGNORECASE):
        return 4
    return None


# 부동소수점 양자화가 '연산 이득'을 내려면 그 폭의 텐서코어가 있어야 한다.
# 없으면 VRAM 만 줄고 속도는 그대로거나(fp8) 아예 커널이 없다(fp4).
FLOAT_QUANT_MIN_CAP = {8: 8.9, 4: 10.0}      # fp8 → SM89(Ada), fp4 → SM100(Blackwell)
FLOAT_QUANT_GEN = {8: "SM89(Ada)/SM90(Hopper)", 4: "SM100(Blackwell)"}


def _looks_like_mlx(owner, repo, config):
    """MLX(Apple Silicon 전용) 배포인가.

    확장자가 .safetensors 인 것은 맞지만 그 안의 양자화 규약이 MLX 고유다.
    safetensors 는 컨테이너일 뿐이고 내용물의 규약은 별개다 — .gguf 라고 전부
    llama.cpp 가 읽는 게 아닌 것과 같다. vLLM 도 PyTorch 도 못 읽는다.

    판정 근거 둘:
      - 이름: mlx-community/* 또는 저장소 이름에 MLX
      - config.json: MLX 는 최상위 "quantization": {"group_size":…, "bits":…} 를 쓴다.
        HF 진영은 "quantization_config" 를 쓰므로 키 이름으로 갈린다.
    """
    if owner.lower() == "mlx-community":
        return True
    if re.search(r"(^|[-_.])mlx([-_.]|$)", repo, re.IGNORECASE):
        return True
    if isinstance(config, dict):
        q = config.get("quantization")
        if isinstance(q, dict) and "bits" in q and "quantization_config" not in config:
            return True
    return False


# quant_method → (판정 기호, 설명). SM80(Ampere) 기준이다.
QUANT_VERDICT = {
    "awq":                 ("◎", "AWQ w4a16 — Marlin 커널. SM80 1순위"),
    "gptq":                ("◎", "GPTQ w4a16/w8a16 — Marlin 커널"),
    "gptq_marlin":         ("◎", "GPTQ (Marlin)"),
    "awq_marlin":          ("◎", "AWQ (Marlin)"),
    "compressed-tensors":  ("◎", "compressed-tensors — vLLM 진영 표준"),
    "marlin":              ("◎", "Marlin"),
    "fp8":                 ("△", "fp8 — SM89(Ada) 이상에만 텐서코어가 있다"),
    "modelopt":            ("△", "ModelOpt(fp8 또는 fp4)"),
    "modelopt_fp4":        ("△", "fp4 — SM100(Blackwell) 미만은 Marlin 가중치 전용으로 폴백"),
    "mxfp4":               ("△", "MXFP4 — SM100 미만은 Marlin 가중치 전용으로 폴백"),
    "bitsandbytes":        ("△", "bitsandbytes — 동작하지만 느리다"),
    "aqlm":                ("△", "AQLM"),
    "gguf":                ("△", "GGUF — vLLM 지원은 실험적. llama.cpp 로 쓰는 편이 낫다"),
    "torchao":             ("△", "torchao"),
}


def gate_repo(owner, repo, revision, files, meta):
    """가중치를 받기 전에 이 저장소가 이 박스에서 쓸 수 있는지 판정한다.

    돌려주는 값: (계속할까 bool, config dict 또는 None)

    GGUF 갈래에는 이 단계가 없다. 있을 이유도 없었다 — 부분 오프로드가 있으니
    일단 받아서 -ngl 로 맞추면 됐다. 여기서는 안 들어가면 그냥 못 쓴다.
    """
    print("")
    print("── 저장소 판정 ──")
    print(f"  {owner}/{repo}  (revision: {revision})")

    cfg_url = hf_file_url(owner, repo, revision, "config.json")
    config = _fetch_json(cfg_url)
    if config is None:
        print("  ⚠ config.json 을 읽지 못했습니다 (gated 이거나 없는 저장소).")
        print("     판정 없이 진행하면 다 받은 뒤에야 문제를 알게 될 수 있습니다.")
        return (prompt("     그래도 계속할까요? (y/N): ") or "N").lower().startswith("y"), None

    blocking = []      # 이게 있으면 기본값이 '중단'
    notes = []

    # ── MLX (문서 §3.3 함정 1위) ──
    is_mlx = _looks_like_mlx(owner, repo, config)
    if is_mlx:
        blocking.append("MLX(Apple Silicon 전용) 배포입니다. NVIDIA GPU 와 무관하며 "
                        "vLLM 도 PyTorch 도 읽지 못합니다.")

    # ── 아키텍처 ──
    archs = config.get("architectures") or []
    if archs:
        print(f"  아키텍처   : {', '.join(archs)}")
    # vLLM 지원 여부는 여기서 단정하지 않는다. 확인하려면 vllm 을 import 해야 하는데
    # (수 초) 이 스크립트는 venv 밖에서도 돌아야 한다. 못 읽으면 기동 시점에 명확한
    # 오류가 나므로, 아는 척하지 않고 그대로 보여주기만 한다.

    # ── 양자화 ──
    qc = config.get("quantization_config") or {}
    method = (qc.get("quant_method") or "").lower()
    if is_mlx:
        # MLX 는 quantization_config 대신 최상위 quantization 을 쓴다. 그걸 안 보고
        # "양자화 없음 = 원본 bf16" 이라고 찍으면 정반대로 안내하게 된다.
        mq = config.get("quantization") if isinstance(config.get("quantization"), dict) else {}
        bits = mq.get("bits")
        print(f"  양자화     : ✗ MLX 고유 양자화{f' ({bits}bit)' if bits else ''} — "
              f"safetensors 컨테이너일 뿐 내용물 규약이 다릅니다")
    elif method:
        mark, desc = QUANT_VERDICT.get(method, ("?", "vLLM 지원 여부 미상"))
        bits = qc.get("bits") or qc.get("weight_bits")
        fbits = _float_quant_bits(method, qc, repo)
        if fbits:
            # 포맷 이름이 아니라 실제 가중치 타입이다. 판정은 이 카드 기준으로 한다.
            bits = fbits
            cap = _gpu_compute_cap()
            need = FLOAT_QUANT_MIN_CAP[fbits]
            gen = FLOAT_QUANT_GEN[fbits]
            if cap and cap >= need:
                mark, desc = "◎", f"fp{fbits} 가중치 — SM{cap * 10:.0f} 에 fp{fbits} 텐서코어가 있습니다"
            elif cap:
                # 못 도는 게 아니다. vLLM 이 Marlin 가중치 전용 경로로 자동 폴백한다
                # (실측: SM80 에서 MarlinNvFp4LinearKernel.is_supported() == True,
                #  fp8 도 w8a8 이 안 되면 w8a16 스킴으로 내려간다).
                # 가중치를 fp16 으로 역양자화해 일반 텐서코어로 계산하므로
                # VRAM·대역폭은 줄지만 연산 이득은 없고, 프리필처럼 연산이 무거운
                # 구간은 오히려 느려질 수 있다 — vLLM 자신이 그렇게 경고를 찍는다.
                mark = "△"
                desc = (f"fp{fbits} 가중치 — 이 GPU(SM{cap * 10:.0f})에 fp{fbits} 텐서코어가 없어 "
                        f"Marlin 가중치 전용으로 폴백합니다. VRAM·대역폭만 이득, "
                        f"연산 이득은 없음 ({gen} 부터 네이티브)")
            else:
                mark = "△"
                desc = f"fp{fbits} 가중치 — {gen} 미만에서는 Marlin 가중치 전용으로 폴백합니다"
            method = f"{method} → fp{fbits}"
        print(f"  양자화     : {mark} {method}{f' ({bits}bit)' if bits else ''} — {desc}")
        if mark == "✗":
            blocking.append(f"양자화 {method}: {desc}. vLLM 이 커널을 찾지 못해 기동에 실패합니다.")
        elif mark == "△":
            notes.append(f"양자화 {method}: {desc}")
    else:
        dt = config.get("torch_dtype") or config.get("dtype") or "?"
        print(f"  양자화     : ○ 없음 (원본 {dt}) — 크기만 맞으면 품질은 최상")

    # ── 멀티모달 ──
    fset = set(files)
    if "preprocessor_config.json" in fset or "video_preprocessor_config.json" in fset:
        kinds = []
        if "preprocessor_config.json" in fset:
            kinds.append("이미지")
        if "video_preprocessor_config.json" in fset:
            kinds.append("영상")
        print(f"  멀티모달   : ● VLM ({'/'.join(kinds)} 입력). 비전 타워가 같은 샤드 안에 "
              f"있으므로 mmproj 를 따로 받지 않습니다")

    # ── 채팅 템플릿 ──
    if "chat_template.jinja" in fset:
        print("  채팅템플릿 : ● chat_template.jinja")
    else:
        print("  채팅템플릿 : ○ 별도 파일 없음 (tokenizer_config.json 안에 있을 수 있음)")

    # ── 원격 코드 ──
    if config.get("auto_map") or any(f.endswith(".py") for f in files):
        notes.append("사용자 정의 모델링 코드가 있습니다 — vLLM 기동 시 "
                     "--trust-remote-code 가 필요할 수 있습니다")

    # ── 크기 vs VRAM ── ★ 이 갈래에만 있는 판정
    keep = filter_repo_files(files)
    weight_b = sum(meta.get(f, (0, ""))[0] for f in keep
                   if f.lower().endswith(WEIGHT_EXTS))
    total_b = sum(meta.get(f, (0, ""))[0] for f in keep)
    print(f"  가중치     : {_fmt_bytes(weight_b)}  (받을 파일 {len(keep)}개, "
          f"합계 {_fmt_bytes(total_b)})")

    vram_mib = _gpu_total_mib()
    if vram_mib and weight_b:
        vram_b = vram_mib * 1024 * 1024
        ratio = weight_b / vram_b
        print(f"  VRAM       : {_fmt_bytes(vram_b)} 중 가중치가 {ratio * 100:.0f}%")
        # KV 캐시·활성화·CUDA 컨텍스트가 들어갈 자리가 남아야 한다. 가중치가 85%를
        # 넘으면 KV 풀이 거의 안 남아 max_model_len 을 의미 있게 잡을 수 없다.
        if ratio >= 1.0:
            blocking.append(f"가중치({_fmt_bytes(weight_b)})가 VRAM({_fmt_bytes(vram_b)})보다 "
                            f"큽니다. vLLM 은 전부 올리지 못하면 기동에 실패합니다.")
        elif ratio > 0.85:
            notes.append(f"가중치가 VRAM 의 {ratio * 100:.0f}% 입니다 — KV 캐시가 거의 "
                         f"남지 않아 컨텍스트를 짧게 잡아야 합니다.")
    elif not vram_mib:
        print("  VRAM       : (nvidia-smi 없음 — 크기 판정을 건너뜁니다)")

    for n in notes:
        print(f"  ⚠ {n}")

    if blocking:
        print("")
        sys.stdout.flush()      # 로그로 리다이렉트하면 stderr 가 먼저 나가 순서가 뒤집힌다
        for b in blocking:
            print(f"  ⛔ {b}", file=sys.stderr)
        sys.stderr.flush()
        print("")
        return (prompt("  그래도 받을까요? (y/N): ") or "N").lower().startswith("y"), config

    return True, config


def filter_repo_files(files):
    """저장소 파일 목록에서 실제로 받을 것만 남긴다.

    HF 저장소에는 vLLM 이 쓰지 않는 파일이 잔뜩 섞여 있다. 대표적으로 같은 가중치의
    구 PyTorch 판(*.bin)이 safetensors 와 나란히 올라와 있는 경우가 흔한데, 안 거르면
    다운로드가 정확히 2배가 된다. GGUF 갈래에는 없던 문제다(파일 하나였으니까).
    """
    low_map = {f: f.lower() for f in files}
    has_st = any(v.endswith(".safetensors") for v in low_map.values())

    keep = []
    for f, low in low_map.items():
        if low.endswith(EXCLUDE_SUFFIX):
            continue
        if any(low.startswith(d) for d in EXCLUDE_DIR_PREFIX):
            continue
        # safetensors 가 있으면 .bin/.pth 는 같은 가중치의 중복이다. 없으면 그게 본체다.
        if has_st and low.endswith((".bin", ".pth", ".pt")):
            continue
        # 그 .bin 들의 샤드 매니페스트도 같이 빠져야 한다. 안 그러면 존재하지 않는
        # 파일을 가리키는 index 만 남아, 나중에 무결성을 확인할 때 혼란만 준다.
        if has_st and low.endswith((".bin.index.json", ".pth.index.json")):
            continue
        # consolidated.* 는 Mistral 계열의 원본 한 덩어리 판. 샤드와 중복이다.
        if os.path.basename(low).startswith("consolidated.") and has_st:
            continue
        keep.append(f)
    return sorted(keep)


def suggest_repo_dirname(owner, repo):
    """<제작자>_<저장소>. GGUF 쪽 suggest_name 과 같은 규칙(제작자 중복은 제거)."""
    return sanitize_name(f"{owner}_{_strip_token(repo, owner)}")


def ask_repo_name(owner, repo):
    suggested = suggest_repo_dirname(owner, repo)
    print("")
    print("저장소에서 뽑은 이름:")
    print(f"  디렉터리 : models/{suggested}/")
    ans = prompt("이 이름을 쓸까요? (Y/n, 또는 원하는 이름을 직접 입력): ")
    low = ans.lower()
    if ans == "" or low in ("y", "yes"):
        return suggested
    if low not in ("n", "no"):
        return sanitize_name(ans.rstrip("/"))
    name = sanitize_name(prompt("디렉터리 이름 (models/<name>/): ").rstrip("/"))
    if not name:
        print("디렉터리 이름이 필요합니다.", file=sys.stderr)
        sys.exit(1)
    return name


def _post_download_report(target_dir, config):
    """받고 나서야 확인할 수 있는 것들. 다음 단계(툴 콜링)의 입력이 된다."""
    print("")
    print("── 받은 저장소 확인 ──")

    # 채팅 템플릿이 도구를 다루는가. 파이프라인의 전제조건이므로 여기서 미리 본다.
    tmpl = ""
    jinja = os.path.join(target_dir, "chat_template.jinja")
    if os.path.isfile(jinja):
        try:
            tmpl = io.open(jinja, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
    if not tmpl:
        tk = os.path.join(target_dir, "tokenizer_config.json")
        if os.path.isfile(tk):
            try:
                with io.open(tk, encoding="utf-8") as f:
                    t = json.load(f).get("chat_template")
                tmpl = t if isinstance(t, str) else ""
            except Exception:
                pass
    if tmpl:
        if re.search(r"\btools?\b", tmpl):
            print("  툴 콜링   : ● 채팅 템플릿이 tools 를 다룹니다")
            print("              → --enable-auto-tool-choice --tool-call-parser <X>")
            print("                파서는 모델 계열에 맞춰야 합니다. 안 맞으면 조용히")
            print("                텍스트로 새어나옵니다 (파서 결정은 run 쪽이 합니다)")
        else:
            print("  툴 콜링   : ○ 채팅 템플릿에 tools 처리가 보이지 않습니다")
    else:
        print("  툴 콜링   : ? 채팅 템플릿을 찾지 못했습니다")

    idx = os.path.join(target_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            with io.open(idx, encoding="utf-8") as f:
                wmap = json.load(f).get("weight_map") or {}
            shards = sorted(set(wmap.values()))
            missing = [x for x in shards if not os.path.isfile(os.path.join(target_dir, x))]
            if missing:
                print(f"  샤드      : ✗ 매니페스트가 요구하는 {len(missing)}개가 없습니다: "
                      f"{', '.join(missing[:3])}{' …' if len(missing) > 3 else ''}",
                      file=sys.stderr)
                return False
            print(f"  샤드      : ✓ 매니페스트의 {len(shards)}개 모두 존재")
        except Exception as exc:
            print(f"  샤드      : ? 매니페스트를 읽지 못했습니다 ({exc})")
    return True


def main_safetensors(owner, repo, revision, files, meta):
    """safetensors 저장소를 통째로 받는다 (vLLM 용)."""
    proceed, config = gate_repo(owner, repo, revision, files, meta)
    if not proceed:
        print("중단했습니다.", file=sys.stderr)
        sys.exit(1)

    keep = filter_repo_files(files)
    if not keep:
        print("받을 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)
    dropped = len(files) - len(keep)

    # 가중치가 gated 인지는 20GB 를 태우고 나서가 아니라 지금 알아야 한다.
    first_weight = next((f for f in keep if f.lower().endswith(WEIGHT_EXTS)), keep[0])
    ensure_token(hf_file_url(owner, repo, revision, first_weight))

    name = ask_repo_name(owner, repo)
    target_dir = os.path.join("models", name)
    os.makedirs(target_dir, exist_ok=True)

    print("")
    print(f"── 받을 파일 ({len(keep)}개"
          f"{f', 제외 {dropped}개' if dropped else ''}) ──")
    jobs, skipped = [], []
    for path in keep:
        total, sha = meta.get(path, (0, ""))
        dest = os.path.join(target_dir, path)          # 하위 디렉터리 구조를 그대로 유지
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = hf_file_url(owner, repo, revision, path)
        job = plan_file(os.path.basename(path)[:28], url, dest, total, sha)
        if job:
            jobs.append(job)
        elif os.path.exists(dest):
            skipped.append((os.path.basename(path), dest, sha))

    if jobs:
        print("")
        # 4개씩. 저장소 하나에 파일이 15개를 넘을 수 있는데 연결만 쪼개면 전부 느려진다.
        if not run_jobs(jobs, max_parallel=4):
            print("", file=sys.stderr)
            print("일부 다운로드가 실패했습니다. 다시 실행하면 받다 만 지점부터 이어받습니다.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print("")
        print("모두 받아져 있습니다 — 받을 것이 없습니다.")

    # ─── 무결성 검증 ───
    # sha 가 있는 것(=LFS, 즉 큰 가중치 샤드)만 해싱한다. 작은 설정 파일은 HF 가
    # oid 를 주지 않아 검증할 수 없지만, 어차피 몇 KB 라 의심되면 다시 받으면 된다.
    to_verify = [(j.label, j.dest, j.sha) for j in jobs]
    if VERIFY_ALL:
        to_verify += skipped
    elif skipped:
        print("")
        print(f"(건너뛴 파일 {len(skipped)}개는 크기만 확인했습니다. "
              f"sha256 까지 보려면 VERIFY_ALL=1 로 실행하세요.)")

    hashable = [t for t in to_verify if t[2]]
    nohash = len(to_verify) - len(hashable)
    if hashable:
        print("")
        print("── 무결성 검증 ──")
        bad = False
        for _label, dest, sha in hashable:
            if not verify_sha(dest, sha):
                bad = True
        if nohash:
            print(f"  (설정 파일 {nohash}개는 HF 가 sha256 을 제공하지 않아 크기만 확인)")
        if bad:
            print("", file=sys.stderr)
            print("파일이 손상되었습니다. 다시 실행하면 이어받기를 시도합니다.", file=sys.stderr)
            print(f"그래도 안 되면 지우고 처음부터 받으세요: rm -rf '{target_dir}'", file=sys.stderr)
            sys.exit(1)

    if not _post_download_report(target_dir, config):
        sys.exit(1)

    print("")
    print("완료. 저장 위치:")
    print(f"  {target_dir}/")
    # ls 는 fd 1 에 직접 쓴다 — 파이썬 버퍼를 우회하므로, 먼저 비워두지 않으면
    # 로그로 리다이렉트했을 때 ls 결과가 앞선 출력보다 먼저 나와 순서가 뒤집힌다.
    sys.stdout.flush()
    subprocess.call(["ls", "-1sh", target_dir])
    print("")
    print("vLLM 은 파일이 아니라 **디렉터리**를 받습니다:")
    print(f"  ./run_vllm_server.py {target_dir}")


def main_gguf(owner, repo, revision, files, meta):
    """GGUF 저장소 (llama.cpp 용). 예전부터의 갈래 — 동작은 그대로다."""
    gguf_files = sorted(f for f in files if f.lower().endswith(".gguf"))
    if not gguf_files:
        print("No GGUF files found in the repository.", file=sys.stderr)
        sys.exit(1)

    model_files = [f for f in gguf_files if "mmproj" not in os.path.basename(f).lower()
                   and not is_mtp_name(f)]
    projection_files = [f for f in gguf_files if "mmproj" in os.path.basename(f).lower()]
    mtp_files = [f for f in gguf_files if is_mtp_name(f)]
    if not model_files and mtp_files:
        # 'Something-MTP.gguf' 처럼 본 모델 파일 이름 자체가 MTP 로 끝나는 저장소도
        # 있을 수 있다. 그런 곳에서 본 모델 목록이 통째로 비어 버리지 않게 되돌린다.
        model_files, mtp_files = mtp_files, []
    if not model_files:
        print("No base model GGUF files found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n저장소: {owner}/{repo} (revision: {revision})")
    print("사용 가능한 양자화 수준:")
    for path in model_files:
        print(f"  - {quantization_label(path)}: {os.path.basename(path)}")
    model_file = choose_file("본 모델", model_files)
    model_url = hf_file_url(owner, repo, revision, model_file)

    proj_file = ""
    if projection_files:
        use_proj = (prompt("프로젝션 모델을 사용할까요? (Y/n): ") or "Y").lower().startswith("y")
        if use_proj:
            proj_file = choose_file("프로젝션 모델", projection_files)
    mtp_file = ""
    if mtp_files:
        use_mtp = (prompt("MTP 모델을 사용할까요? (y/N): ") or "N").lower().startswith("y")
        if use_mtp:
            mtp_file = choose_file("MTP 모델", mtp_files)

    proj_url = hf_file_url(owner, repo, revision, proj_file) if proj_file else ""
    mtp_url = hf_file_url(owner, repo, revision, mtp_file) if mtp_file else ""
    ensure_token(model_url)

    model_dir_name = ask_names(model_url, proj_url, "-MTP" if mtp_file else "")
    target_dir = os.path.join("models", model_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    plans = [("model", model_file, model_url,
              os.path.join(target_dir, f"{model_dir_name}.gguf"))]
    if proj_url:
        plans.append(("mmproj", proj_file, proj_url, os.path.join(
            target_dir, f"{model_dir_name}_{proj_tag(url_basename(proj_url))}.gguf")))
    if mtp_url:
        plans.append(("mtp", mtp_file, mtp_url,
                      os.path.join(target_dir, f"{model_dir_name}_mtp.gguf")))

    print("")
    print("── 받을 파일 ──")
    jobs, skipped = [], []
    for label, path, url, dest in plans:
        total, sha = meta.get(path, (0, ""))
        job = plan_file(label, url, dest, total, sha)
        if job:
            jobs.append(job)
        elif os.path.exists(dest):
            skipped.append((label, dest, sha))

    if jobs:
        print("")
        ok = run_jobs(jobs)
        if not ok:
            print("", file=sys.stderr)
            print("일부 다운로드가 실패했습니다. 다시 실행하면 받다 만 지점부터 이어받습니다.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print("")
        print("모두 받아져 있습니다 — 받을 것이 없습니다.")

    # ─── 무결성 검증 ───
    # 이번에 받은(또는 이어받은) 파일만 검사한다. 이미 있던 파일까지 매번 다시
    # 해싱하면 30GB 당 70초 정도를 아무 일 없이 태우게 된다. 전부 확인하고 싶으면
    # VERIFY_ALL=1 로 실행한다.
    to_verify = [(j.label, j.dest, j.sha) for j in jobs]
    if VERIFY_ALL:
        to_verify += skipped
    elif skipped:
        print("")
        print(f"(건너뛴 파일 {len(skipped)}개는 크기만 확인했습니다. "
              f"sha256 까지 보려면 VERIFY_ALL=1 로 실행하세요.)")

    if to_verify:
        print("")
        print("── 무결성 검증 ──")
        bad = False
        for _label, dest, sha in to_verify:
            if not verify_sha(dest, sha):
                bad = True
        if bad:
            print("", file=sys.stderr)
            print("파일이 손상되었습니다. 다시 실행하면 이어받기를 시도합니다.", file=sys.stderr)
            print(f"그래도 안 되면 지우고 처음부터 받으세요: rm '{target_dir}'/*.gguf",
                  file=sys.stderr)
            sys.exit(1)

    print("")
    print("Downloads complete. Saved to:")
    # ls 는 fd 1 에 직접 쓴다 — 파이썬 버퍼를 우회하므로, 먼저 비워두지 않으면
    # 로그로 리다이렉트했을 때 ls 결과가 앞선 출력보다 먼저 나와 순서가 뒤집힌다.
    sys.stdout.flush()
    subprocess.call(["ls", "-1sh", target_dir])


def main():
    """저장소 URL 하나만 받아서 갈래는 스스로 정한다.

    사용자가 "이건 llama.cpp 용인가 vLLM 용인가" 를 미리 알아야 할 이유가 없다.
    트리 API 를 한 번 부르면 어느 쪽인지 파일 목록에 이미 답이 있다.
    """
    repo_url = sys.argv[1] if len(sys.argv) > 1 else prompt("Hugging Face model repository URL: ")
    if not repo_url:
        print("Repository URL is required.", file=sys.stderr)
        sys.exit(1)

    owner, repo, revision, files, meta = list_repo_tree(repo_url)
    kind = detect_repo_kind(files)

    if kind == "both":
        print("")
        print(f"저장소 {owner}/{repo} 에 두 형식이 다 있습니다.")
        print("  [1] safetensors — vLLM 용 (저장소 전체를 받습니다)")
        print("  [2] GGUF        — llama.cpp 용 (파일을 골라 받습니다)")
        kind = "gguf" if (prompt("번호 선택 [기본값 1]: ") or "1").strip() == "2" else "safetensors"

    if kind == "safetensors":
        return main_safetensors(owner, repo, revision, files, meta)
    if kind == "gguf":
        return main_gguf(owner, repo, revision, files, meta)

    print(f"저장소 {owner}/{repo} 에서 모델 가중치를 찾지 못했습니다 "
          f"(safetensors 도 GGUF 도 없음).", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
