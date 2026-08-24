# vLLM 학습 계획 — llama.cpp 를 아는 사람을 위한

*작성 2026-08-23 · 대상: 언락 CMP 170HX (GA100 / SM80) 박스*
*필독 전제: [`~/Developments/170hx_maintenance/AGENTS.md`](../../170hx_maintenance/AGENTS.md)*
*상위 맥락: [`~/Developments/imageai/docs/agent_pipeline_plan.md`](../../imageai/docs/agent_pipeline_plan.md) — 이 학습은 그 계획의 Phase 0·4 에 해당한다*

---

## 0. 이 문서의 목적

llama.cpp 는 익숙하고 vLLM 은 처음이다. **이미 아는 것에 대응시켜 배우는 것**이 가장 빠르다. 그래서 이 문서는 vLLM 튜토리얼이 아니라 **llama.cpp ↔ vLLM 번역 사전 + 실측 커리큘럼**이다.

**학습 완료 기준 (이게 되면 "vLLM 을 안다"고 해도 된다):**

1. 모델을 띄우고 VRAM·기동시간·지연을 **의도한 대로 조절**할 수 있다
2. **툴 콜링이 성공**한다 (파이프라인의 전제조건)
3. llama.cpp 와 **어느 조건에서 누가 이기는지** 실측으로 안다
4. 안 되는 것(GGUF·부분 오프로드·MLX)을 **미리 알아본다**

> **왜 이걸 먼저 하는가:** vLLM 설치의 8할은 vLLM 이 아니라 **CUDA 파이썬 환경(py3.12 + torch cu12x)** 을 세우는 일이고, 그 환경은 **ComfyUI 의 전제와 동일**하다. vLLM 을 끝내면 ComfyUI 가 절반 끝난 상태로 시작된다.

---

## 1. 이 박스의 전제

| 사실 | vLLM 에 미치는 영향 |
|:---|:---|
| **실사용 VRAM ≈ 63.4 GiB** (`nvidia-smi` 의 65536 MiB 는 라벨) | §1.1 산수 참조. **clamp 덕에 초과 할당은 깨끗한 OOM** — 실험해도 카드는 안 죽는다 |
| **PCIe Gen2 x4** (≈1.6GB/s) | **기동이 느리다.** 가중치 적재 + CUDA 그래프 캡처로 수 분. 학습 중엔 `--enforce-eager` 로 줄인다 |
| BF16 ≈161 TFLOPS, SM 70 | 연산은 A100 급. **병목은 연산이 아니라 전송** |
| **SM80 → fp8 텐서코어 없음** | fp8 양자화 이득 0. **AWQ / GPTQ-INT4 (Marlin 커널)** 로 간다 |
| 시스템 RAM 30GB | 적재 경로가 RAM 을 탄다. 대형 모델 **동시 기동 금지** |
| Python 3.14 (시스템·conda 둘 다) | **torch/vLLM 휠이 없다. py3.12 venv 필수** |
| ⛔ **apt 로 kernel / `nvidia-*` 업그레이드 금지** | **언락 소멸.** CUDA 유저스페이스는 **pip/uv 로만** |
| 드라이버 610.43.02 (최신) | 오히려 유리 — 휠이 요구하는 최소 드라이버를 전부 만족 |
| 전력캡 180W 상주 | 지속부하 시 HBM 85°C 상한 확인 |

### 1.1 `--gpu-memory-utilization` 산수 (정정)

**GPU 전체(65536 MiB) 대비 비율**이다. 실사용 상한과의 관계:

| util | 계산값 | 판정 |
|---:|---:|:---|
| 0.90 | 57.6 GiB | 안전 |
| 0.95 | 60.8 GiB | **안전** (실사용 63.4 GiB 아래) |
| 0.98 | 62.7 GiB | 빠듯하지만 가능 |
| **0.9906** | **63.4 GiB** | **물리적 천장** |

> 초기 메모에 "0.95 를 주면 안 된다"고 썼던 것은 **틀렸다.** cmpunlocker 의 clamp(`limit 0x0000000F7FFFFFFF`)가 존재하지 않는 구간을 PMA 에서 제외하므로, 넘겨도 Xid 하드폴트가 아니라 **CUDA OOM** 으로 깨끗이 실패한다. 위험선은 0.99 근처다.

**진짜 주의할 것은 다른 프로세스다.** util 은 *전체* 기준이라, llama-server 가 19GB(0.29)를 쥔 채로 vLLM 에 0.9 를 주면 합이 넘친다.

```
util ≈ (vLLM 에 주고 싶은 GiB) / 64
# llama.cpp(19GB) 와 공존 → vLLM 은 util ≤ 0.65
# vLLM 단독 → 0.90~0.95
```

---

## 2. llama.cpp → vLLM 번역 사전

### 2.1 실행 옵션

| llama.cpp | vLLM | 주의 |
|:---|:---|:---|
| `-m model.gguf` | `vllm serve <HF리포 또는 디렉터리>` | **파일이 아니라 디렉터리를 준다** (§3) |
| `-ngl 99` | **대응물 없음** | ⚠️ **vLLM 은 부분 오프로드를 안 한다.** 전부 GPU 아니면 실패 |
| `-c 32768` | `--max-model-len 32768` | llama.cpp 는 딱 그만큼 잡지만, **vLLM 은 남은 VRAM 전부를 KV 풀로 선점**한다 |
| (VRAM 자동) | `--gpu-memory-utilization 0.9` | §1.1 |
| `--parallel 4` | **기본 동작** | 연속 배칭이 vLLM 의 존재 이유. 따로 켤 것 없음 |
| `-b` / `-ub` | `--max-num-batched-tokens` | 처리량/지연 트레이드오프 손잡이 |
| (동시 요청 상한) | `--max-num-seqs` | |
| `--jinja` (툴콜) | `--enable-auto-tool-choice --tool-call-parser <X>` | **최대 사고 지점.** 모델별로 파서가 다르다 |
| `--chat-template` | `--chat-template` | 동일 개념 |
| `-fa` | 기본 (FlashAttention) | SM80 에서 FA2 동작 |
| MTP 드래프터 | `--speculative-config '{"method":"mtp",...}'` | §3.4 |
| (없음) | **`--enforce-eager`** | CUDA 그래프 캡처를 꺼서 **기동 대폭 단축.** 학습 중엔 켤 것 |
| `--host/--port` | `--host/--port` | 동일 |
| OpenAI 호환 API | 동일 | **기존 습관 그대로 통한다** |

### 2.2 사고방식의 차이

| | llama.cpp | vLLM |
|:---|:---|:---|
| 설계 목표 | **아무 데서나 돌아가게** (CPU/부분 오프로드/온갖 백엔드) | **GPU 를 최대로 짜내기** |
| 최적 지점 | 1인 1스트림 | **동시 다수 요청** |
| 메모리 | 필요한 만큼 | **미리 크게 잡고 페이지로 관리**(PagedAttention) |
| 기동 | 수십 초 | **수 분** |
| 모델 포맷 | GGUF 단일 파일 | HF 디렉터리 |

**결론:** 지금의 1인 채팅에서는 vLLM 이 체감상 안 빨라진다. **툴 루프·프롬프트 정제·VLM 검수로 한 턴에 LLM 을 여러 번 동시에 부를 때** 갚는다. 지금은 미래 투자 + 환경 학습으로 볼 것.

---

## 3. 모델 파일 — GGUF 와 무엇이 다른가

### 3.1 왜 GGUF 는 1개고 safetensors 는 여러 개인가

| | GGUF | HF safetensors |
|:---|:---|:---|
| 철학 | **전부 한 파일에** — 가중치 + 토크나이저 + 메타데이터 + 채팅템플릿 | **텐서만.** 나머지는 옆의 JSON 들 |
| 파일 수 | 1개 (크면 `-00001-of-0000N` 분할) | **샤드 여러 개 + 설정 파일 다수** |
| 넘기는 것 | 파일 경로 | **디렉터리 / 리포 이름** |

safetensors 가 ~5GB 단위로 쪼개지는 건 **업로드·다운로드·mmap 효율 관례**다. 어떤 텐서가 어느 샤드에 있는지는 **`model.safetensors.index.json`** 이 들고 있다(스크린샷의 218 kB 짜리). 로더는 이 매니페스트를 읽어 알아서 조립하므로, **사용자는 샤드를 신경 쓸 필요가 없다.**

**한 리포의 구성 요소:**

| 파일 | 역할 | GGUF 에선 |
|:---|:---|:---|
| `model-0000N-of-0000M.safetensors` | 가중치 샤드 | 본체에 포함 |
| `model.safetensors.index.json` | 샤드 매니페스트 | 불필요 |
| `config.json` | 아키텍처 정의 | 메타데이터로 포함 |
| `tokenizer.json` / `tokenizer_config.json` | 토크나이저 | 포함 |
| `chat_template.jinja` | 채팅 템플릿 | 포함 |
| `generation_config.json` | 샘플링 기본값 | 일부 포함 |
| `preprocessor_config.json` | **이미지 전처리 설정** | `mmproj` 쪽 |

### 3.2 프로젝션 모델(mmproj)은 어디로 갔나

**HF 리포에는 별도 파일이 없다. 비전 타워와 프로젝터가 같은 샤드 안에 들어 있다.**

llama.cpp 가 `mmproj-*.gguf` 를 따로 두는 건 GGUF 가 "한 파일 = 한 모델" 구조라 멀티모달을 억지로 붙였기 때문이다. HF 는 원래부터 하나의 체크포인트에 전부 담는다. `preprocessor_config.json` 이 있으면 그 리포는 멀티모달이고, vLLM 은 **아무것도 추가로 지정하지 않아도** 통째로 올린다.

> 스크린샷 리포에 `preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json` 이 있다 → 그 모델은 이미지·영상 입력을 받는 VLM 이다.

### 3.3 ⚠️ 스크린샷의 함정 — MLX 는 이 박스에서 못 쓴다

`Qwen3.8-27B-Uncensored-**MLX**` + `2-bit/4-bit/6-bit/8-bit` 폴더 구성은 **MLX(Apple Silicon 전용 프레임워크)** 배포의 전형이다.

**확장자가 `.safetensors` 인 것은 맞지만, 그 안의 양자화 방식이 MLX 고유 포맷이다.** safetensors 는 "컨테이너"일 뿐이고 내용물의 규약은 별개다 — GGUF 확장자라고 다 llama.cpp 가 읽는 게 아닌 것과 같다. **vLLM 도 PyTorch 도 못 읽는다. NVIDIA GPU 와 무관한 물건이다.**

**받아야 할 것을 고르는 법:**

| 리포 이름/태그 | 판정 |
|:---|:---|
| `-MLX`, `mlx-community/*` | ✗ **Apple 전용** |
| `-AWQ`, `-GPTQ`, `-Int4`, `compressed-tensors` | ◎ **이것들** |
| `-FP8`, `fp8_scaled` | △ SM80 에 연산기 없음 → 이득 0 |
| `-GGUF` | △ vLLM 지원은 있으나 실험적. **llama.cpp 로 쓸 것** |
| 태그 없는 원본 (fp16/bf16) | ○ 크기만 맞으면 최상 품질 |

### 3.4 MTP 는 어떻게 되나

vLLM 에 **네이티브 MTP 지원이 있다.** 별도 드래프트 모델 없이 타깃 모델의 MTP 헤드를 쓴다.

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

**단, "vLLM 이 그 모델 계열의 MTP 를 구현했을 때"만 된다.** 공식 문서가 명시한 계열:

- **Gemma 4 (E2B / E4B / 12B / 26B-A4B / 31B IT)** — 별도 assistant 체크포인트를 지정
  ```
  --speculative-config '{"method":"mtp","model":"<assistant-ckpt>","num_speculative_tokens":1}'
  ```
- `XiaomiMiMo/MiMo-7B-Base`

> **손에 있는 모델과의 관계:** `HauhauCS_Gemma4-26B-A4B-...-MTP` / `Gemma4-31B-...-MTP` 는 **vLLM 이 MTP 를 지원하는 계열**이다(단 safetensors AWQ/원본 + assistant 체크포인트를 새로 받아야 한다). `Qwen3.8` 계열은 목록에 없다 → MTP 없이 쓰거나 EAGLE/드래프트 모델 방식으로.

---

## 4. 양자화 포맷 선택 (SM80 기준)

| 포맷 | SM80 | 비고 |
|:---|:---:|:---|
| **AWQ (w4a16)** | ◎ | LLM·VLM 전부. **1순위** |
| **GPTQ (w4a16 / w8a16)** | ◎ | Marlin 커널로 빠르다 |
| **compressed-tensors (w4a16 / w8a8-int8)** | ◎ | vLLM 진영 표준. 신형 모델에 많다 |
| FP8 | ✗ 이득 없음 | Ada/Hopper 부터 |
| GGUF | △ | 지원은 있으나 실험적·아키텍처 제한. **`Q8_K_P` 같은 비표준 양자화는 못 읽는다** |
| MLX | ✗ | Apple 전용 |
| bitsandbytes | △ | 되지만 느리다 |

**첫 모델 선정 기준:**

1. **작을 것 (8B급).** 27B 를 붙들고 배우면 실험 1회에 10분씩 날아간다. 다운로드 5~6GB, 기동 1분대여야 손잡이를 100번 돌려볼 수 있다
2. **AWQ 또는 GPTQ-INT4**
3. **툴 콜링 지원 + 파서가 알려진 계열** (Qwen 계열 등)
4. 한국어

> 구체적 HF 리포 이름은 **실행 시점에 검색해서 정한다.** 존재하지 않는 이름을 추측하지 않는다.

---

## 5. 커리큘럼

각 단계는 **완료 판정**을 넘기면 끝낸다. 넘어가기 전에 실측을 `bench/` 에 남긴다.

### Step 0 — 환경 (반나절)
```bash
uv venv --python 3.12 --seed ~/Developments/vllm/.venv
source ~/Developments/vllm/.venv/bin/activate
uv pip install vllm --torch-backend=auto     # ⛔ apt 로 nvidia-* 설치 금지
```
- 스모크: `torch.cuda.get_device_capability() == (8,0)`, bf16 matmul, **62 GiB 부근에서 깨끗한 OOM** 확인
- 작업 후 `sudo ~/Developments/170hx_maintenance/cmpunlocker/verify.sh`

**완료 판정:** `python -c "import vllm; print(vllm.__version__)"` + 스모크 통과 + verify 정상.

### Step 1 — 첫 기동 (§4 기준으로 8B급 AWQ)
```bash
vllm serve <repo> --port 8000 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 8192 \
  --enforce-eager                  # 학습 중엔 켠다: 기동 시간 단축
```
- llama-server 가 떠 있으면 util 을 낮추거나 먼저 내린다 (§1.1)
- `curl :8000/v1/models`, `/v1/chat/completions` — **llama-server 와 같은 모양이라 습관이 그대로 통한다**

**완료 판정:** 응답이 오고, `nvidia-smi` 의 점유가 util 계산값과 맞아떨어지는 이유를 설명할 수 있다.

### Step 2 — 손잡이 실측 ★핵심
바꿔가며 **기동시간 / peak VRAM / 첫토큰지연 / 토큰속도**를 기록한다.

| 손잡이 | 관찰할 것 |
|:---|:---|
| `--gpu-memory-utilization` 0.3 / 0.5 / 0.9 | KV 풀 크기 → **동시 처리 가능 시퀀스 수**가 어떻게 변하나 |
| `--max-model-len` 4k / 32k / 128k | 왜 늘리면 KV 가 급증하나 |
| `--enforce-eager` on/off | **기동시간 vs 추론속도** 트레이드오프 (PCIe Gen2 x4 의 대가 실측) |
| `--max-num-seqs`, `--max-num-batched-tokens` | 처리량 vs 지연 |

**완료 판정:** "동시 8명을 32k 컨텍스트로 받으려면 util 을 얼마로?" 에 계산으로 답할 수 있다.

#### 실측 결과 (2026-08-24, Qwen3.8-27B, **전력캡 180W**)

> **측정 조건 주의:** 아래 표는 전부 **180W** 에서 잰 값이다 (`cmp-maintain.service` 가
> 부팅마다 180W 를 재적용하고, 이날 17:17 재부팅이 있었다). 250W 로 올리면 연산 관련
> 수치가 약 14% 높아진다 — [`quantization_concepts.md`](quantization_concepts.md) §2 참조.
> **서로 비교하는 데는 문제가 없다. 전부 같은 조건이다.**

| 설정 | 디코드 | TTFT | 프리필 | 동시16 | VRAM | 기동 |
|:---|---:|---:|---:|---:|---:|---:|
| A · W8A8 · eager · util0.94 | 14.6 tok/s | 643 ms | **222 µs/자** | 139.3 t/s | — | — |
| B · AWQ · eager · util0.94 | 14.4 tok/s | 178 ms | 337 µs/자 | 211.6 t/s | 60,298 | **88 s** |
| C · AWQ · **no-eager** · util0.94 | **54.7 tok/s** | 129 ms | 299 µs/자 | **561.9 t/s** | 60,246 | 271 s |
| **D · AWQ · no-eager · util0.50 · KV fp8** | **54.3 tok/s** | **57 ms** | 304 µs/자 | **551.7 t/s** | **31,182** | 252 s |

**★ `--enforce-eager` 를 끄면 디코드가 3.80배 빨라진다** (14.4 → 54.7 tok/s).
대가는 기동 88초 → 271초 (`torch.compile` 만 77.8초). **상주 서버라면 1회성이므로
`--no-eager` 가 정답이다.** `--enforce-eager` 는 손잡이를 자주 돌리는 학습 단계 전용이다.

**★ 양자화가 구간을 가른다** — 같은 27B 모델의 두 판을 같은 조건으로:
- 프리필: W8A8 222 vs AWQ 337 µs/자 → **W8A8 이 1.52배 빠름** (A축 이득)
- 동시 배치: AWQ 211.6 vs W8A8 139.3 t/s → **AWQ 가 1.52배 빠름** (W축 이득)

[`quantization_concepts.md`](quantization_concepts.md) §5.1 의 예측이 실제 모델에서 확인됐다.

**★ C → D: VRAM 을 절반으로 줄여도 성능은 그대로다**

| | C (util 0.94) | D (util 0.50 + fp8 KV) | 유지율 |
|:---|---:|---:|---:|
| 디코드 | 54.7 tok/s | 54.3 tok/s | **99.3%** |
| 동시16 | 561.9 t/s | 551.7 t/s | **98.2%** |
| 프리필 | 299 µs/자 | 304 µs/자 | 98.4% |
| **VRAM** | 60,246 MiB | **31,182 MiB** | **51.8%** |

**fp8 KV 가 토큰 밀도를 1.97배로 올려 util 절반의 손해를 되찾는다**
(KV 풀 38.45→9.80 GiB 인데 토큰은 622,785→312,733). 직접 증거: vLLM 이 어텐션 블록
크기를 **784 → 1,568 토큰으로 정확히 2배** 늘렸다.

> **이 구성이 실전 답이다** — ComfyUI 등에 **33.5 GiB** 를 내주고도 성능 손실이 1~2% 다.
> 상세: [`~/Developments/llm_interfaces/docs/interface_plan.md`](../../llm_interfaces/docs/interface_plan.md) §2

**★ 연속 배칭은 거의 선형으로 확장한다** — 동시 1→16 에서 14.2 → 211.6 t/s (**14.93배**).
이것이 vLLM 을 쓰는 이유 그 자체다.

#### ★ 손잡이는 독립이 아니다 — `util` ↓ 이면 `--max-num-seqs` 도 ↓

`util 0.50` 으로 내렸더니 기동이 이 오류로 실패했다:

```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (212).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
```

**Qwen3.8 은 하이브리드 모델**이다 — 순수 트랜스포머가 아니라 GDN(선형 어텐션) 층이 섞여 있다.
기동 로그가 이미 말해준다:

```
Using Triton/FLA GDN prefill kernel
Setting attention block size to 784 tokens to ensure that attention page size is >= mamba page size
```

그래서 **디코드 시퀀스마다 Mamba 캐시 블록이 1개씩** 필요한데, 이 블록 수는 KV 예산에 비례한다.
`util` 을 반으로 줄이면 블록도 반으로 주는데 `--max-num-seqs` 기본값(256)은 그대로라 충돌한다.

> **규칙: `--max-num-seqs` ≤ (로그가 알려주는 Mamba 블록 수).**
> 하이브리드 모델을 낮은 util 로 띄울 때 반드시 같이 조정해야 한다.
> `run_vllm_server.py` 가 이 오류를 감지해 안내한다.

### Step 3 — 툴 콜링 ★파이프라인의 전제조건
```bash
vllm serve <repo> --enable-auto-tool-choice --tool-call-parser <모델에 맞는 파서>
```
- `tools=[...]` 를 넣은 요청으로 `tool_calls` 가 **파싱되어** 오는지 확인
- **파서 불일치가 최대 사고 유형.** 인자가 문자열로 뭉개져 오거나 아예 텍스트로 새어나온다
- 실패 시 텍스트 폴백을 어떻게 감지할지 정해둔다 (오케스트레이터 설계 입력)

**완료 판정:** 2개 이상 도구를 준 상태에서 올바른 도구·올바른 인자로 10회 중 9회 이상 성공.

### Step 4 — llama.cpp 대조 벤치
같은 급 모델로:

| 시나리오 | 측정 |
|:---|:---|
| 1스트림 짧은 프롬프트 | TTFT, tok/s |
| 1스트림 긴 컨텍스트(16k+) | TTFT, tok/s |
| **동시 4·8스트림** | **총 처리량, 지연 분포** ← vLLM 이 이기는 지점 |
| 기동 | 콜드 스타트 시간 |
| VRAM | peak |

**완료 판정:** "언제 vLLM 을 쓰고 언제 llama.cpp 를 쓸지" 를 **수치로** 말할 수 있다.

#### 실측 결과 (2026-08-24, 180W)

**대조쌍:** Qwen3.8-27B · 양쪽 8비트 (vLLM `W8A8 int8` 34.0 GiB ↔ llama.cpp `Q8_0` 29.3 GiB)
· 입력 ~4,200 토큰 · 출력 150 토큰 고정(`ignore_eos`+`min_tokens`)

| 동시 | vLLM W8A8 | | | llama.cpp Q8 | | | **vLLM 우위** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| | 페이지/분 | TTFT | 배수 | 페이지/분 | TTFT | 배수 | |
| 1 | 9.8 | 1,601 ms | 1.00x | 5.8 | 5,076 ms | 1.00x | **1.69x** |
| 4 | 22.2 | 4,234 ms | 2.27x | 10.3 | 12,257 ms | 1.79x | **2.16x** |
| 8 | 27.9 | 7,309 ms | 2.85x | 13.9 | 21,061 ms | 2.41x | **2.01x** |
| 16 | **32.2** | 13,441 ms | **3.28x** | 15.9 | 43,282 ms | 2.77x | **2.03x** |

| | vLLM | llama.cpp |
|:---|---:|---:|
| VRAM | 58,330 MiB (util 0.90) | 39,284 MiB |

**① vLLM 이 전 구간 우세하고 동시성이 오를수록 격차가 벌어진다** (1.69 → 2.03배).

**② TTFT 격차가 더 크다** — 동시 16 에서 13.4초 vs 43.3초로 **3.2배**.

**③ ★ 진짜 차이는 처리량이 아니라 메모리 모델이다.**

llama-server 기동에 **세 번 실패했고 전부 같은 원인**이었다:

| `-c` | `--parallel` | 슬롯당 | 결과 |
|---:|---:|---:|:---|
| 32,768 | 16 | 2,048 | ✗ `request (2056 tokens) exceeds 2048` |
| 65,536 | 16 | 4,096 | ✗ `request (4198 tokens) exceeds 4096` |
| 131,072 | 16 | 8,192 | ✓ |

**vLLM 에서는 한 번도 없었다.**

| | llama.cpp | vLLM |
|:---|:---|:---|
| 동시성 | **`--parallel N` 사전 선언** | 동적 |
| 컨텍스트 | **N 등분 고정** — 슬롯보다 큰 요청은 **거부** | 공유 KV 풀에서 필요한 만큼 |
| 긴 입력 × 높은 동시성 | **둘의 곱만큼 `-c`** → VRAM 비례 증가 | 추가 비용 없음 |

> llama.cpp 로 "동시 16 × 8k 컨텍스트" 를 하려면 `-c 131,072` 가 필요했다.
> vLLM 은 같은 일을 `GPU KV 캐시` 하나로 처리한다.

#### 결론 — 언제 무엇을 쓰나

| 상황 | 선택 | 근거 |
|:---|:---|:---|
| **동시 요청 · 긴 입력** (만화 배치, 문서 처리) | **vLLM** | 처리량 2배 · TTFT 3.2배 · 슬롯 제약 없음 |
| **1인 1스트림** | vLLM 1.69배 우세하나 격차가 가장 작다 | 이 구간은 llama.cpp 도 쓸 만하다 |
| **VRAM 이 빠듯하다** | **llama.cpp** | 같은 8비트에서 39GB vs 58GB. `-ngl` 로 부분 오프로드도 가능 |
| **GGUF 만 있는 모델** | **llama.cpp** | 형식 지원이 넓다 |
| 기동을 자주 한다 | llama.cpp | vLLM 은 콜드 271초 (캐시 히트 106초) |

> ⚠️ **대조쌍 선정에 두 번 실패했다.** Gemma4-26B 로 4비트끼리 맞추려 했으나
> vLLM 이 그 AWQ 저장소를 못 읽었다
> (`AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute`).
> Qwen3.8 은 vLLM 이 int4, llama.cpp 가 Q8 이라 비트가 어긋나서,
> **vLLM 쪽을 W8A8 로 바꿔야 8비트끼리 맞았다.**

### Step 5 — 대형 모델 / MTP (선택)
- 27B급 AWQ 로 확장, util 재계산
- **Gemma 4 계열이면 MTP 실측** (§3.4) — llama.cpp 의 MTP 와 속도 비교
- 멀티모달(VLM) 리포를 올려보고 `preprocessor_config.json` 의 역할 확인 (§3.2)

---

## 6. 실측 기록 양식

`~/Developments/vllm/bench/runs.tsv` (imageai/llamacpp 의 TSV 관례 재사용)

```
date  model  quant  util  max_model_len  enforce_eager  concurrency  startup_s  ttft_ms  tok_s  peak_vram_mib  note
```

---

## 7. 함정 체크리스트

1. ⛔ **`apt` 로 kernel / `nvidia-*` 를 올리지 않는다** — 언락 소멸. CUDA 는 pip/uv 로만
2. **Python 3.14 로 시도하지 않는다** — py3.12 venv
3. **MLX 리포를 받지 않는다** (§3.3)
4. **손에 있는 GGUF 는 vLLM 에서 못 쓴다** — `Q8_K_P`/`MTP` 는 llama.cpp 전용. 새로 받아야 한다
5. **`-ngl` 같은 부분 오프로드가 없다** — 안 들어가면 그냥 OOM
6. **util 은 전체 기준** — 다른 프로세스 점유를 빼서 계산 (§1.1)
7. **fp8 리포는 무의미** — SM80 에 연산기 없음
8. **기동이 느린 건 고장이 아니다** — PCIe Gen2 x4. `--enforce-eager` 로 완화
9. **OOM 은 정상 동작** — clamp 가 하드폴트를 막아준다. 겁내지 말고 실험할 것
10. **툴콜 파서를 모델에 맞춘다** — 안 맞으면 조용히 텍스트로 샌다

---

## 8. 참고

- vLLM 문서: https://docs.vllm.ai/
- 양자화 지원 하드웨어: https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html
- MTP: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- 투기적 디코딩 전반: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- 박스 환경(필독): `~/Developments/170hx_maintenance/AGENTS.md`
- 상위 파이프라인 계획: `~/Developments/imageai/docs/agent_pipeline_plan.md`
