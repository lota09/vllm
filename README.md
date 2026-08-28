# vLLM on CMP 170HX

메모리 언락된 **NVIDIA CMP 170HX(64 GiB HBM2e · SM80)** 에서 vLLM 을 돌리기 위한
프로비저닝·다운로드·기동 도구와, 그 과정에서 나온 **실측 기록**.

llama.cpp 를 쓰던 사람이 vLLM 으로 넘어오면서 부딪힌 것들을 그때그때 도구로 만들고
문서로 남긴 저장소다. 범용 튜토리얼이 아니라 **이 박스 기준**이다.

## 이 하드웨어의 특이점

| | |
|:---|:---|
| VRAM | **64 GiB HBM2e** (스톡 8 GiB 에서 언락) · 대역폭 1,592.8 GB/s |
| 연산 | SM80(Ampere) — **fp8·fp4 텐서코어 없음**. int8 303 TOPS · BF16 168 TFLOPS |
| PCIe | **Gen2 x4** (~1.5 GB/s) — 적재는 느리지만 디코드에는 무관 |
| 전력 | 기본 캡 250W, 최대 300W. 상시 운용은 **180W 권장** |

⛔ `apt` 로 kernel / `nvidia-*` 업그레이드 금지 — 언락이 소멸한다. CUDA 는 pip/uv 로만.
자세한 것은 [`~/Developments/170hx_maintenance/AGENTS.md`](../170hx_maintenance/AGENTS.md).

## 도구

### `setup_vllm.sh`
uv 기반 프로비저닝. conda 를 PATH 에서 걷어내고 파이썬 3.12 venv 를 만든 뒤
vLLM 을 설치하고 스모크 테스트까지 돌린다. GPU·드라이버·디스크 점검 포함.

### `download_model.py`
받기 **전에** 이 카드에서 쓸 수 있는지 판정한다. llamacpp 저장소와 하드링크로 공유한다.

```
── 저장소 판정 ──
  아키텍처   : Qwen3_5ForConditionalGeneration
  양자화     : ◎ compressed-tensors — vLLM 진영 표준
  W축(가중치): int4 · group 128 — SM80 네이티브
  A축(활성화): 없음 (weight-only) — SM80 에서 손해 없음
  가중치     : 18.2 GiB   VRAM: 64.0 GiB 중 28%
```

- **W축·A축을 `config.json` 에서 읽는다.** 저장소 이름은 W축만 말하거나 틀리게 말한다
- **디렉터리 이름을 규칙대로 제안한다** — `<제작자>_<베이스>_<특성>_<W축>-<A축>`
- MLX·GGUF 오인, 고아 샤드 중복 계산, gated 저장소를 걸러낸다
- gated 면 그 자리에서 토큰을 받아 `secrets/huggingface_token.json` 에 저장한다

### `run_vllm_server.py`
모델을 고르고 util 을 계산해 vLLM 을 띄운다. 대화형이 기본이고 옵션으로 전부 건너뛴다.

```bash
./run_vllm_server.py                              # 전부 물어봄
./run_vllm_server.py models/<name> --util 0.90 -y # 비대화형
```

**기본값 두 개가 실측에서 정해졌다:**

| 손잡이 | 기본 | 왜 |
|:---|:---|:---|
| CUDA 그래프 | **켜짐** (`--eager` 로 끔) | 디코드 **3.8배**. 대가는 기동 88→271초인데 캐시 히트면 106초 |
| `--kv` | **auto** | 안전한 쪽. fp8 은 이득이 큰 모델에만 권한다 |

기존 서버 종료 여부·포트·util·KV dtype 을 묻고, 기동 로그를 감시해 치명적 오류를 판별한다.
`--kv` 기본값은 **auto**(모델 dtype 그대로) — 안전한 쪽이 기본이다.
`--kv fp8` 은 토큰 밀도를 1.97배로 만들지만 조건이 붙는다(아래).

### `bench/` (로컬 전용, 아래 참조)
| 도구 | 재는 것 |
|:---|:---|
| `bench_manga.py` | 워크로드 모양대로 (입력·출력 길이 고정, 동시성 스윕) |
| `bench_accuracy.py` | perplexity · top-1 적중률 · **긴 문맥 PPL** · 정답 50문 |
| `bench_power.py` | 워크로드 크기별 전력·클럭·효율 |
| `bench_decode_pl.py` | 전력캡별 정상상태 디코드 속도 (TTFT 제외) |
| `bench_divergence.py` | 두 설정의 출력이 토큰 단위로 언제 갈리나 |
| `bench_quality.py` | needle 검색 · 반복 붕괴 · 언어 혼입 |

## 문서

- **[`docs/quantization.md`](docs/quantization.md)** — 세 축(W/A/KV) · 형식별 호환성 ·
  폴백이 일어나는 이유 · 정확도 손실 실측 · 저장소 고르는 법
- **[`docs/hardware.md`](docs/hardware.md)** — 이 카드의 연산 유닛 · 클럭 도메인 ·
  전력·발열 · 전력캡이 성능에 미치는 영향
- **[`docs/performance.md`](docs/performance.md)** — 프리필/디코드 역학 ·
  **CUDA 그래프 vs eager** · 연속 배칭 · KV 축 · llama.cpp 대조
- **[`docs/vllm_learning_plan.md`](docs/vllm_learning_plan.md)** — llama.cpp ↔ vLLM 번역 사전과
  단계별 커리큘럼
- `models/model_bookmark.md` — 후보 모델을 `W축-A축` 으로 분류한 목록 (로컬 전용, 아래 참조)

## 주요 실측 결과

**양자화 축 — 이 카드에서 A축은 이득이 없다**

| | SM80 |
|:---|:---|
| W축 (int4/fp4/fp8/int8 무엇이든) | ✓ 전부 Marlin 가중치 전용으로 처리 — **자유롭다** |
| A축 int8 | ✓ **유일한 실연산 이득** — 프리필 TTFT 1.14~1.16배 |
| A축 fp8 · fp4 | △ 폴백되어 무시된다 — **정확도만 잃으므로 순손실** |

`get_min_capability()` 가 능력을 요구해도 **거부가 아니라 폴백**이다.
A축 때문에 기동이 막히는 조합은 (아는 한) 없다.

**속도 — Qwen3.8-27B AWQ(w4a16) · 동시 1 디코드**

| 설정 | tok/s |
|:---|---:|
| `--enforce-eager` · 180W | 14.40 |
| CUDA 그래프 · 180W | 53.72 |
| CUDA 그래프 · 300W | **62.09** |

`--enforce-eager` 를 끄면 **3.73배**, 전력캡을 300W 로 올리면 추가 15.6%.
대역폭 활용률 76.2% 로 커뮤니티가 말하는 170HX 범위(60~70 tok/s) 안에 든다.

**품질 — fp8 KV 는 공짜다**

같은 가중치에서 KV dtype 만 바꿔 재보면 64k 문맥까지 PPL 비율 0.987~0.999,
정답 과제 43/50 동일. **출력은 바뀌지만 품질은 안 바뀐다.**

## 측정하며 배운 것

값을 세 번 뒤집은 원인들이다. 벤치를 짤 때 이것부터 막는 게 낫다.

| 함정 | 증상 |
|:---|:---|
| 출력 길이 미고정 | 두 모델이 40 vs 16 토큰 생성 → 처리량 비교 무의미 |
| 같은 서버에 벤치 2개 | 디코드 54.3 → 12.4 tok/s |
| 모델 라벨 오인 | 기동 실패했는데 **이전 서버가 살아남아** 측정됨 |
| health 200 조기 통과 | 새 서버 대기 중 **옛 서버**의 200 을 보고 통과 |
| 설정을 안 보고 하드웨어를 의심 | `--enforce-eager` 서버를 재고 카드를 의심했다 |

> **하드웨어를 의심하기 전에 무엇을 재고 있는지부터 확인하라.**
> `/proc/<pid>/cmdline` 을 읽는 데 1초면 된다.

## 추적하지 않는 것

`.gitignore` 가 `models/` `bench/` `logs/` `secrets/` `.venv` 를 제외한다.
모델 가중치와 로그를 빼려는 것인데, 그 결과 **벤치 도구와 모델 북마크도 같이 빠진다.**
필요하면 예외를 넣을 것:

```gitignore
bench/*
!bench/*.py
!bench/*.sh
models/*
!models/*.md
```

`secrets/` 는 HuggingFace 토큰이 들어가므로 **반드시 제외 상태를 유지한다.**

## 요구사항

- NVIDIA GPU (이 저장소는 SM80 기준) · CUDA 드라이버
- `uv` (setup 스크립트가 설치한다)
- 파이썬 3.12
