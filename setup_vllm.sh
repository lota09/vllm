#!/usr/bin/env bash
set -euo pipefail
# vLLM 환경 프로비저닝 + 검증 스크립트 (CUDA 전용)
#
# ─────────────────────────────────────────────────────────────
# build_llama_server.sh 와의 관계 — 왜 "build" 가 아니라 "setup" 인가
# ─────────────────────────────────────────────────────────────
# llama.cpp 는 소스만 배포하므로 사용자가 nvcc 로 직접 컴파일해야 한다.
# 그래서 build_llama_server.sh 는 cmake 탐색 / 백엔드 감지 / CUDA arch 결정 /
# host gcc 호환성 / ldd 기반 .so 수집 같은 빌드 관련 로직이 대부분을 차지한다.
#
# vLLM 은 정반대다. PyPI 에 manylinux 바이너리 휠로 배포되고, CUDA 커널이
# 이미 fatbin 으로 컴파일되어 휠 안에 들어있다. nvcc 도 cmake 도 필요 없다.
#   → 빌드 단계가 통째로 사라진다.
#   → 대신 위험이 "설치는 됐는데 GPU 를 못 잡음" 쪽으로 이동한다. 조용히 실패한다.
# 그래서 이 스크립트의 무게중심은 빌드가 아니라 §6 검증에 있다.
#
# ─────────────────────────────────────────────────────────────
# 설계 목표
# ─────────────────────────────────────────────────────────────
#   - CUDA(NVIDIA) 환경만 가정한다. ROCm/Metal/CPU 분기 없음
#   - 멱등(idempotent): 몇 번을 돌려도 안전하고, 2회차부터는 사실상 환경 헬스체크
#   - sudo 를 한 번도 쓰지 않는다 (전부 $HOME 안에서 끝난다)
#   - ⛔ apt 를 한 번도 호출하지 않는다 — 이 박스에서 kernel/nvidia-* 를 건드리면
#     CMP 언락이 소멸한다. CUDA 유저스페이스는 pip/uv 휠로만 조달한다
#   - 실패해도 어디서 왜 실패했는지 요약에 남긴다 (첫 실패에서 죽지 않는다)
#
# 사용:
#   ./setup_vllm.sh
#   VLLM_VERSION=0.11.0 ./setup_vllm.sh      # 버전 핀
#   FORCE_RECREATE=1 ./setup_vllm.sh         # venv 재생성
#   SKIP_SMOKE=1 ./setup_vllm.sh             # 설치만

# ─────────────────────────────────────────────
# 설정 (환경변수로 오버라이드 가능)
# ─────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR=${VENV_DIR:-$SCRIPT_DIR/.venv}
PY_VERSION=${PY_VERSION:-3.12}
VLLM_VERSION=${VLLM_VERSION:-}          # 비우면 최신
TORCH_BACKEND=${TORCH_BACKEND:-auto}    # auto | cu126 | cu128 ...
FORCE_RECREATE=${FORCE_RECREATE:-0}
SKIP_SMOKE=${SKIP_SMOKE:-0}
SKIP_OOM_PROBE=${SKIP_OOM_PROBE:-0}
NONINTERACTIVE=${NONINTERACTIVE:-0}
MIN_FREE_GB=${MIN_FREE_GB:-15}          # venv 가 torch+CUDA 런타임으로 10GB+ 를 먹는다
UV_INSTALLER_URL=${UV_INSTALLER_URL:-https://astral.sh/uv/install.sh}

# 최종 요약에 모을 결과
declare -a WARNINGS=()
SMOKE_FAILS=0

section() { echo ""; echo "── $* ──"; }
warn()    { echo "  [경고] $*"; WARNINGS+=("$*"); }

# URL 을 stdout 으로 받아온다. curl 이 없는 최소 설치에서도 동작하도록 wget 폴백.
# (build_llama_server.sh 와 동일한 헬퍼 — 의도적으로 같은 모양을 유지한다)
fetch_stdout() {
  if   command -v curl >/dev/null 2>&1; then curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
  else return 1
  fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vLLM 환경 프로비저닝 (CUDA 전용)"
echo "  venv   : $VENV_DIR"
echo "  python : $PY_VERSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─────────────────────────────────────────────
# 0. 가드 & 환경 중화
# ─────────────────────────────────────────────
section "0. 환경 중화"

if [ "$(id -u)" -eq 0 ]; then
  warn "root 로 실행 중입니다. uv 와 venv 가 /root 아래에 설치됩니다. 일반 사용자로 실행하는 것을 권장합니다."
fi

# conda 를 PATH 에서 제거하는 이유:
#   miniconda base 가 활성화된 상태에서 venv 를 만들면 PATH/LD_LIBRARY_PATH 에
#   conda 의 python·libstdc++·CUDA 계열 .so 가 섞여 들어온다. 그러면
#   "venv 안의 torch 가 conda 의 libcudart 를 물어버리는" 종류의 사고가 나고,
#   증상이 import 시점이 아니라 커널 실행 시점에 터져서 추적이 매우 어렵다.
#
#   ⚠️ conda 자체를 지우거나 망가뜨리지 않는다. 이 스크립트 프로세스의 PATH 에서만
#   빼는 것이고, build_llama_server.sh 는 여전히 conda 의 nvcc 를 써야 한다
#   (vLLM 은 휠이라 nvcc 가 필요 없지만, llama.cpp 빌드에는 필요하다).
_strip_conda_from_path() {
  local roots=() r p out=()
  [ -n "${CONDA_PREFIX:-}" ] && roots+=("$CONDA_PREFIX")
  # ⚠️ 이 함수 안에서는 외부 명령을 쓰지 않는다 (순수 bash 확장만).
  # 들어온 PATH 가 conda 경로만으로 이루어진 셸이 실제로 존재하는데
  # (conda activate 직후의 최소 환경), 거기엔 coreutils 가 없어서
  # dirname 조차 "command not found" 로 죽는다 — PATH 를 고치러 온 함수가
  # PATH 때문에 죽는 상황이 된다. ${VAR%/*/*} 로 /bin/conda 두 단계를 잘라낸다.
  [ -n "${CONDA_EXE:-}" ]    && roots+=("${CONDA_EXE%/*/*}")
  for r in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    [ -d "$r" ] && roots+=("$r")
  done
  [ ${#roots[@]} -eq 0 ] && return 0

  local IFS=:
  for p in $PATH; do
    local hit=0
    for r in "${roots[@]}"; do
      case "$p" in "$r"/*|"$r") hit=1; break;; esac
    done
    [ "$hit" -eq 0 ] && out+=("$p")
  done

  # conda 를 지웠더니 PATH 가 비어버리는 경우가 실제로 있다.
  # conda 설치 시 PATH 앞에 붙는 경로만 남은 셸(예: conda activate 직후 최소 환경)에서
  # 그렇게 되고, 그러면 이후 모든 외부 명령이 "command not found" 로 죽는다.
  # 시스템 기본 경로로 폴백해서 스크립트가 계속 진행할 수 있게 한다.
  if [ ${#out[@]} -eq 0 ]; then
    PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"
    export PATH
    echo "  conda 제외 후 PATH 가 비어 시스템 기본 경로로 폴백했습니다"
    return 0
  fi

  PATH="${out[*]}"
  export PATH
  return 0
}

if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "conda 활성 감지: $CONDA_PREFIX → 이 프로세스의 PATH 에서만 제외합니다 (conda 는 그대로 둡니다)"
else
  echo "conda 비활성 — PATH 정리만 수행"
fi
_strip_conda_from_path
# 인터프리터가 conda 쪽 표준 라이브러리를 보게 만드는 변수들. venv 에 치명적이다.
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
hash -r 2>/dev/null || true
echo "PATH 정리 완료"

# ─────────────────────────────────────────────
# 1. GPU / 드라이버 전제 확인
# ─────────────────────────────────────────────
section "1. GPU / 드라이버 확인"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  [실패] nvidia-smi 를 찾을 수 없습니다."
  echo "         vLLM 은 부분 오프로드가 없어 GPU 없이는 의미가 없습니다."
  echo "         ⛔ apt 로 nvidia-* 를 설치하지 마십시오 (언락 소멸). 드라이버는 별도 절차로 처리합니다."
  exit 1
fi

GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
              --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
if [ -z "$GPU_INFO" ]; then
  echo "  [실패] nvidia-smi 는 있으나 GPU 정보를 읽지 못했습니다. 드라이버 상태를 확인하십시오."
  exit 1
fi

IFS=',' read -r GPU_NAME DRIVER_VER COMPUTE_CAP VRAM_TOTAL_MIB <<<"$GPU_INFO"
# csv 필드에 붙는 공백 제거
GPU_NAME="$(echo "$GPU_NAME" | xargs)"
DRIVER_VER="$(echo "$DRIVER_VER" | xargs)"
COMPUTE_CAP="$(echo "$COMPUTE_CAP" | xargs)"
VRAM_TOTAL_MIB="$(echo "$VRAM_TOTAL_MIB" | xargs)"

echo "  GPU        : $GPU_NAME"
echo "  드라이버   : $DRIVER_VER"
echo "  Compute Cap: $COMPUTE_CAP"
echo "  VRAM(라벨) : ${VRAM_TOTAL_MIB} MiB"

case "$COMPUTE_CAP" in
  8.0) echo "  → SM80 (Ampere/GA100). AWQ·GPTQ-INT4 Marlin 커널 사용 가능. fp8 텐서코어는 없습니다." ;;
  8.9|9.*|10.*|12.*) echo "  → fp8 텐서코어 보유. fp8 양자화가 유효합니다." ;;
  7.*|8.*) echo "  → fp8 텐서코어 없음. AWQ/GPTQ-INT4 로 가십시오." ;;
  *) warn "예상 밖의 compute capability ($COMPUTE_CAP). vLLM 지원 여부를 확인하십시오." ;;
esac

# 드라이버 메이저 버전으로 대략적인 하한만 본다.
# 정확한 요구치는 torch 휠의 CUDA 버전에 달려있고, forward-compat 도 있어서
# 여기서 단정하지 않는다. §6 의 torch.cuda.is_available() 이 진짜 판정이다.
DRIVER_MAJOR="${DRIVER_VER%%.*}"
if [ "${DRIVER_MAJOR:-0}" -lt 525 ] 2>/dev/null; then
  warn "드라이버 $DRIVER_VER 는 최신 CUDA 12.x 휠에 낮을 수 있습니다. §6 스모크에서 확인됩니다."
fi

# ─────────────────────────────────────────────
# 2. uv 확보
# ─────────────────────────────────────────────
section "2. uv 확보"

# uv 를 쓰는 이유:
#   ① 시스템 파이썬이 3.14 여도 3.12 를 자동으로 받아온다.
#      python -m venv 는 "이미 설치된 인터프리터"만 쓸 수 있어 여기서 탈락한다.
#   ② --torch-backend=auto 로 torch 의 CUDA 인덱스(cu126/cu128/...)를 자동 선택한다.
#   ③ torch+CUDA 런타임은 수 GB 다. 병렬 다운로드 + 글로벌 캐시 + 하드링크로
#      재시도가 거의 공짜가 된다 — 환경을 여러 번 뒤엎는 학습 단계에서 중요하다.
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
# 현재 프로세스에서 즉시 쓸 수 있도록 후보 경로를 미리 얹는다 (설치 전/후 모두 커버).
PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH
hash -r 2>/dev/null || true

if command -v uv >/dev/null 2>&1; then
  echo "  uv 발견: $(command -v uv) ($(uv --version 2>/dev/null))"
else
  echo "  uv 없음 → 설치합니다 (단일 정적 바이너리, 시스템 파이썬을 건드리지 않습니다)"
  if ! fetch_stdout "$UV_INSTALLER_URL" | sh; then
    echo "  [실패] uv 설치 실패. curl/wget 및 네트워크를 확인하십시오."
    exit 1
  fi
  hash -r 2>/dev/null || true
  if ! command -v uv >/dev/null 2>&1; then
    echo "  [실패] uv 설치는 됐으나 PATH 에서 찾지 못했습니다: $UV_INSTALL_DIR"
    exit 1
  fi
  echo "  uv 설치 완료: $(command -v uv) ($(uv --version 2>/dev/null))"
fi

# --torch-backend 지원 여부 확인.
# 없으면 오래된 uv 이므로 수동 인덱스로 폴백한다 (조용히 틀린 휠을 받는 것보다 낫다).
UV_HAS_TORCH_BACKEND=0
if uv pip install --help 2>/dev/null | grep -q -- '--torch-backend'; then
  UV_HAS_TORCH_BACKEND=1
else
  warn "이 uv 버전은 --torch-backend 를 지원하지 않습니다 (uv self update 권장). PyPI 기본 torch 휠로 진행합니다."
fi

# ─────────────────────────────────────────────
# 3. python venv
# ─────────────────────────────────────────────
section "3. python ${PY_VERSION} venv"

_venv_pyver() {
  [ -x "$VENV_DIR/bin/python" ] || return 1
  "$VENV_DIR/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null
}

NEED_CREATE=1
if [ -d "$VENV_DIR" ]; then
  CUR_PY="$(_venv_pyver || true)"
  if [ "$FORCE_RECREATE" = "1" ]; then
    echo "  FORCE_RECREATE=1 → 기존 venv 를 재생성합니다 (현재: ${CUR_PY:-불명})"
  elif [ "$CUR_PY" = "$PY_VERSION" ]; then
    echo "  기존 venv 재사용 (python $CUR_PY)"
    NEED_CREATE=0
  else
    echo "  기존 venv 의 python 이 ${CUR_PY:-불명} 입니다 (요구: $PY_VERSION) → 재생성"
  fi

  if [ "$NEED_CREATE" = "1" ]; then
    # 기존 설치를 지우지 않고 옆으로 밀어둔다. 새 환경이 깨졌을 때 되돌릴 수 있어야 한다.
    BACKUP_DIR="${VENV_DIR}_backup_$(date +%s)"
    echo "  기존 venv 백업: $BACKUP_DIR"
    mv "$VENV_DIR" "$BACKUP_DIR"
  fi
fi

if [ "$NEED_CREATE" = "1" ]; then
  # --seed: pip/setuptools 를 venv 안에 넣어둔다. uv 없이도 그 venv 를 다룰 수 있어야
  #         나중에 락인 없이 다른 도구로 갈아탈 수 있다.
  echo "  생성 중... (시스템에 python $PY_VERSION 이 없으면 uv 가 자동으로 받아옵니다)"
  uv venv --python "$PY_VERSION" --seed "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || { echo "  [실패] venv 생성 실패: $VENV_PY 없음"; exit 1; }
echo "  python: $("$VENV_PY" --version 2>&1) @ $VENV_PY"

# ─────────────────────────────────────────────
# 4. 디스크 여유 확인
# ─────────────────────────────────────────────
section "4. 디스크 여유 확인"

# torch + nvidia-* CUDA 런타임 휠 + vLLM 으로 venv 가 10GB 를 넘는다.
# 설치 도중 ENOSPC 로 죽으면 venv 가 반쯤 깨진 채로 남아 진단이 어려워지므로 미리 막는다.
FREE_KB="$(df -Pk "$VENV_DIR" 2>/dev/null | awk 'NR==2{print $4}')"
if [ -n "${FREE_KB:-}" ]; then
  FREE_GB=$(( FREE_KB / 1024 / 1024 ))
  echo "  여유 공간: ${FREE_GB} GiB (필요 추정: ${MIN_FREE_GB} GiB)"
  if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "  [실패] 공간이 부족합니다. uv 캐시 정리(uv cache clean) 또는 다른 경로(VENV_DIR=)를 쓰십시오."
    exit 1
  fi
else
  warn "디스크 여유 공간을 확인하지 못했습니다. 계속 진행합니다."
fi

# ─────────────────────────────────────────────
# 5. vLLM 설치
# ─────────────────────────────────────────────
section "5. vLLM 설치"

_installed_vllm_ver() {
  "$VENV_PY" -c 'import importlib.metadata as m;print(m.version("vllm"))' 2>/dev/null
}

CUR_VLLM="$(_installed_vllm_ver || true)"
SPEC="vllm"
[ -n "$VLLM_VERSION" ] && SPEC="vllm==$VLLM_VERSION"

if [ -n "$CUR_VLLM" ] && { [ -z "$VLLM_VERSION" ] || [ "$CUR_VLLM" = "$VLLM_VERSION" ]; } ; then
  echo "  이미 설치됨: vllm $CUR_VLLM → 설치 단계 건너뜀"
  echo "  (최신으로 올리려면: VLLM_VERSION=<버전> 또는 uv pip install -p $VENV_PY -U vllm)"
else
  INSTALL_ARGS=(pip install --python "$VENV_PY" "$SPEC")
  if [ "$UV_HAS_TORCH_BACKEND" = "1" ]; then
    INSTALL_ARGS+=("--torch-backend=$TORCH_BACKEND")
    echo "  torch backend: $TORCH_BACKEND (드라이버를 보고 CUDA 인덱스를 자동 선택합니다)"
  fi
  echo "  실행: uv ${INSTALL_ARGS[*]}"
  echo "  (수 GB 다운로드 — 최초 1회는 시간이 걸립니다. 재시도는 uv 캐시 덕에 빠릅니다)"
  uv "${INSTALL_ARGS[@]}"
  CUR_VLLM="$(_installed_vllm_ver || echo '불명')"
  echo "  설치 완료: vllm $CUR_VLLM"
fi

# ─────────────────────────────────────────────
# 6. 스모크 테스트  ★ 이 스크립트의 핵심
# ─────────────────────────────────────────────
section "6. 스모크 테스트"

if [ "$SKIP_SMOKE" = "1" ]; then
  echo "  SKIP_SMOKE=1 → 건너뜀"
else
  # 각 검사는 실패해도 다음으로 넘어간다. 파이썬 종료코드 = 실패 개수.
  # "설치는 성공했는데 GPU 를 못 잡는" 조용한 실패를 잡는 것이 목적이다.
  set +e
  SKIP_OOM_PROBE="$SKIP_OOM_PROBE" "$VENV_PY" - <<'PYEOF'
import os, sys

fails = 0
def check(name, fn):
    global fails
    try:
        msg = fn()
        print(f"  [OK] {name}: {msg}")
    except Exception as e:
        fails += 1
        print(f"  [!!] {name}: {type(e).__name__}: {e}")

import_ok = {}

def t_torch():
    import torch
    import_ok['torch'] = torch
    return f"torch {torch.__version__} (CUDA {torch.version.cuda})"
check("torch import", t_torch)

torch = import_ok.get('torch')

def t_avail():
    if torch is None: raise RuntimeError("torch 미로드")
    if not torch.cuda.is_available():
        # 여기가 가장 흔한 조용한 실패 지점이다.
        raise RuntimeError("torch.cuda.is_available() == False — 드라이버와 휠의 CUDA 버전 불일치 가능")
    return f"{torch.cuda.device_count()}개 디바이스: {torch.cuda.get_device_name(0)}"
check("CUDA 사용 가능", t_avail)

def t_cap():
    cap = torch.cuda.get_device_capability()
    note = "SM80 — AWQ/GPTQ-INT4 Marlin 경로" if cap == (8, 0) else ""
    return f"compute capability {cap[0]}.{cap[1]} {note}".strip()
check("compute capability", t_cap)

def t_bf16():
    # 텐서코어 bf16 경로가 실제로 도는지. import 만으로는 알 수 없다.
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 미지원")
    a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    c = (a @ b).float()
    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    tol = 2.0  # bf16 은 가수 8비트라 1024 누적에서 이 정도 오차는 정상
    if not (err < tol):
        raise RuntimeError(f"bf16 matmul 오차 과다: {err}")
    torch.cuda.synchronize()
    return f"1024x1024 bf16 matmul 정상 (max abs err {err:.3f})"
check("bf16 matmul", t_bf16)

def t_oom():
    # 문서 §1.1 검증: 물리 상한을 넘겨도 Xid 하드폴트가 아니라 깨끗한 CUDA OOM 으로
    # 떨어지는가. 이게 확인돼야 --gpu-memory-utilization 을 겁내지 않고 실험할 수 있다.
    if os.environ.get("SKIP_OOM_PROBE") == "1":
        raise RuntimeError("SKIP_OOM_PROBE=1 로 건너뜀")
    chunks, gib = [], 0
    try:
        while gib < 512:
            chunks.append(torch.empty(1024**3, dtype=torch.uint8, device='cuda'))
            gib += 1
    except torch.cuda.OutOfMemoryError:
        result = f"{gib} GiB 에서 깨끗한 CUDA OOM (하드폴트 없음 — 정상)"
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        result = f"{gib} GiB 에서 깨끗한 OOM (하드폴트 없음 — 정상)"
    else:
        result = f"{gib} GiB 까지 할당했으나 OOM 미발생"
    finally:
        del chunks
        torch.cuda.empty_cache()
    return result
check("점진 할당 → 깨끗한 OOM", t_oom)

def t_vllm():
    import vllm
    return f"vllm {vllm.__version__}"
check("vllm import", t_vllm)

sys.exit(fails)
PYEOF
  SMOKE_FAILS=$?
  set -e
  if [ "$SMOKE_FAILS" -eq 0 ]; then
    echo "  모든 검사 통과"
  else
    warn "스모크 테스트 ${SMOKE_FAILS}건 실패 (위 [!!] 항목 참조)"
  fi
fi

# ─────────────────────────────────────────────
# 7. --gpu-memory-utilization 계산기
# ─────────────────────────────────────────────
section "7. --gpu-memory-utilization 권장값"

# 문서 §1.1 / §7-6 의 함정을 스크립트가 대신 기억한다:
#   util 은 "GPU 전체" 대비 비율이지 "남은 VRAM" 대비가 아니다.
#   llama-server 가 19GB 를 쥔 채로 vLLM 에 0.9 를 주면 합이 넘친다.
OTHER_MIB=0
while read -r line; do
  [ -z "$line" ] && continue
  m="$(echo "$line" | awk -F, '{gsub(/ /,"",$2); print $2}')"
  case "$m" in ''|*[!0-9]*) continue;; esac
  OTHER_MIB=$(( OTHER_MIB + m ))
done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)

# 물리 상한. CMP 170HX 는 nvidia-smi 가 65536 MiB 를 보고하지만 실사용은 약 63.4 GiB 다
# (cmpunlocker 의 clamp 가 존재하지 않는 구간을 PMA 에서 제외한다).
# 다른 카드에서는 라벨값을 그대로 쓴다.
case "$GPU_NAME" in
  *"CMP 170HX"*) PHYS_MIB=${PHYS_VRAM_MIB:-64921} ;;
  *)             PHYS_MIB=${PHYS_VRAM_MIB:-$VRAM_TOTAL_MIB} ;;
esac

AVAIL_MIB=$(( PHYS_MIB - OTHER_MIB ))
[ "$AVAIL_MIB" -lt 0 ] && AVAIL_MIB=0

echo "  VRAM 라벨      : ${VRAM_TOTAL_MIB} MiB  (util 의 분모)"
if [ "$PHYS_MIB" != "$VRAM_TOTAL_MIB" ]; then
  echo "  물리 상한      : ${PHYS_MIB} MiB  ← 라벨보다 작습니다 (clamp)"
fi
echo "  타 프로세스    : ${OTHER_MIB} MiB"
echo "  vLLM 가용      : ${AVAIL_MIB} MiB"

if [ "$VRAM_TOTAL_MIB" -gt 0 ]; then
  UTIL_MAX="$(awk -v a="$AVAIL_MIB" -v t="$VRAM_TOTAL_MIB" 'BEGIN{printf "%.2f", a/t}')"
  # 안전 마진 0.05. 물리 상한 근처는 CUDA 컨텍스트/단편화로 실패하기 쉽다.
  UTIL_REC="$(awk -v u="$UTIL_MAX" 'BEGIN{v=u-0.05; if(v>0.95)v=0.95; if(v<0)v=0; printf "%.2f", v}')"
  echo ""
  echo "  → 상한 util    : $UTIL_MAX  (여기를 넘기면 OOM)"
  echo "  → 권장 util    : $UTIL_REC"
  echo "     vllm serve <repo> --gpu-memory-utilization $UTIL_REC"
  if [ "$OTHER_MIB" -gt 0 ]; then
    echo "     ⚠ 타 프로세스가 ${OTHER_MIB} MiB 를 쥐고 있어 그만큼 낮춰 계산했습니다."
    echo "       그 프로세스를 내리면 다시 실행해 값을 갱신하십시오."
  fi
fi

# ─────────────────────────────────────────────
# 8. 언락 상태 검증 안내
# ─────────────────────────────────────────────
VERIFY_SH="$HOME/Developments/170hx_maintenance/cmpunlocker/verify.sh"
if [ -f "$VERIFY_SH" ]; then
  section "8. 언락 검증"
  # sudo 가 필요하므로 자동 실행하지 않는다. 이 스크립트는 sudo 를 쓰지 않는다는
  # 원칙을 지키고, 사용자가 직접 판단해서 돌리게 한다.
  echo "  작업 후 언락 상태를 확인하십시오:"
  echo "    sudo $VERIFY_SH"
fi

# ─────────────────────────────────────────────
# 9. 완료 요약
# ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$SMOKE_FAILS" -eq 0 ] && [ "${#WARNINGS[@]}" -eq 0 ]; then
  echo " 설정 완료 — 이상 없음"
elif [ "$SMOKE_FAILS" -eq 0 ]; then
  echo " 설정 완료 (경고 ${#WARNINGS[@]}건)"
else
  echo " 설정 완료 — 검증 실패 ${SMOKE_FAILS}건. 아래를 확인하십시오"
fi
echo ""
echo "  GPU        : $GPU_NAME (SM ${COMPUTE_CAP}, 드라이버 $DRIVER_VER)"
echo "  venv       : $VENV_DIR"
echo "  python     : $("$VENV_PY" --version 2>&1)"
TORCH_LINE="$("$VENV_PY" -c 'import torch;print(f"{torch.__version__} / CUDA {torch.version.cuda}")' 2>/dev/null || echo '확인 실패')"
echo "  torch      : $TORCH_LINE"
echo "  vllm       : ${CUR_VLLM:-확인 실패}"

if [ "${#WARNINGS[@]}" -gt 0 ]; then
  echo ""
  echo "  경고:"
  for w in "${WARNINGS[@]}"; do echo "    - $w"; done
fi

echo ""
echo "  활성화:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  첫 기동 (문서 Step 1 — 8B급 AWQ 로 시작):"
echo "    vllm serve <HF리포> --port 8000 \\"
echo "      --gpu-memory-utilization ${UTIL_REC:-0.35} \\"
echo "      --max-model-len 8192 \\"
echo "      --enforce-eager"
echo ""
echo "  이 스크립트는 멱등입니다 — 다시 실행하면 수 초 만에 끝나는 환경 헬스체크가 됩니다."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$SMOKE_FAILS"
