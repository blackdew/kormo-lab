# kormo-lab

KORMo(Korean Open Reasoning Model) 기반 한국어 LLM from-scratch 학습 실습 프로젝트.

## 구성

```
kormo-lab/
├── KORMo-tutorial/   # MLP-Lab/KORMo-tutorial 클론 (외부 코드, git 추적 제외)
│   ├── tutorial/     # 01.pretrain_from_scratch → 02.sft_qlora → 03.inference
│   ├── src/kormo/    # KORMo 아키텍처 구현 + KORMoTrainer
│   └── src/scripts/  # 실전용 accelerate 실행 스크립트 (1B/10B)
├── notebooks/
│   ├── 01_kormo_1B_pretrain_colab_a100.ipynb  # Colab A100용 사전학습 노트북 (수정본)
│   ├── 02_kormo_eval_colab.ipynb              # 벤치마크 평가 노트북 (L4로 충분)
│   └── 03_kormo_vl_mini_colab.ipynb           # 미니 KORMo-VL — 1B 백본 + SigLIP 프로젝터 정렬
├── scripts/
│   └── pretrain_local.py   # 로컬 Mac(MPS)용 사전학습 스크립트 — 노트북 파이프라인의 이식판
├── .venv/            # Python 3.12 (uv), git 추적 제외
└── README.md
```

## Colab에서 학습 실행

`notebooks/01_kormo_1B_pretrain_colab_a100.ipynb`를 Colab에 업로드 → 런타임 유형 A100 → 모두 실행.

원본 튜토리얼 노트북 대비 변경점:
- uv·flash-attn 설치 제거 (flex_attention 사용이라 불필요 — 빌드 시간 30분+ 절약)
- `num_proc` 하드코딩(48/128) → `os.cpu_count()` 자동 설정
- transformers 4.57.0(yanked) → 4.57.1
- Google Drive 저장 기반 **3-모드 자동 분기**:
  - **fresh**: Drive에 아무것도 없으면 새로 초기화해 학습
  - **resume**: `checkpoint-*`만 있으면(학습 중 끊김) optimizer 상태까지 복원해 정확히 재개
  - **continue**: `final/`이 있으면 학습된 가중치를 로드해 그 위에 추가 학습
  - `FORCE_FRESH = True`로 기존 결과를 타임스탬프 백업 폴더로 옮기고 처음부터 재학습 가능
- 저장 위치는 `MyDrive/kormo-1B-PT/output/` (체크포인트·`final/`), 평가 이력은 `evals/`에 분리.
  구버전 레이아웃(`kormo-1B-PT/` 바로 아래 `final/`)은 실행 시 자동으로 `output/`으로 이동
- 체크포인트는 optimizer 포함(~8GB) 최근 1개 유지, 완료 시 `output/final/`(2.6GB)에 배포용 저장 — Drive 무료 15GB 내
- **wandb 로깅 기본 활성화** (`USE_WANDB`) — 프로젝트 `kormo-lab`, run 이름 `kormo-1B-{모드}`.
  API 키는 Colab 보안 비밀 `WANDB_API_KEY`에서 읽음 (없으면 로그인 프롬프트)
- **검증셋 분리 + 학습 중 평가** — 고정 seed로 128 시퀀스(~0.5M 토큰)를 떼어 100 스텝마다
  `eval/loss`·`eval/mean_token_accuracy` 로깅. multi-epoch 학습에서 train loss는 2 epoch째부터
  암기 효과로 낙관적이므로 검증 지표가 진짜 진행 기준
- **모드별 학습률 분기** — fresh/resume 5e-4, continue 5e-5. 수렴한 가중치에 from-scratch용
  peak LR을 그대로 쓰면 train loss가 상승하며 성능이 파괴됨

## 평가 (`02_kormo_eval_colab.ipynb`)

Drive의 `output/final/` 모델을 lm-evaluation-harness(0.4.12)로 평가. L4 GPU면 충분.

1. **Held-out PPL** — 한국어 위키 200문서. 튜토리얼 스케일에서 가장 민감한 진행 지표
2. **KoBEST + HAE-RAE** — loglikelihood 채점이라 작은 base 모델도 측정 가능 (0-shot, ~10분)
3. **KMMLU** — `RUN_KMMLU=True`로 활성화 (5-shot, KORMo 논문 설정, 1시간+)
4. **결과 시각화** — 누적된 `evals/eval_*.json`으로 최신 평가 vs 랜덤 기준선 바 차트,
   평가 이력 PPL(로그 스케일)·정확도 추이 차트. 평가 재실행 없이 이 섹션만 단독 실행 가능

결과는 `MyDrive/kormo-1B-PT/evals/eval_*.json`에 누적 — 추가 학습 반복 시 성능 곡선 비교용

## 환경 (uv)

이미 세팅된 `.venv`가 있으면 활성화만 하면 됩니다:

```bash
source .venv/bin/activate
```

처음부터 다시 만들려면 (uv 기준):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch "transformers==4.57.1" "datasets>=4.1.1" accelerate pyyaml
# kormo 패키지를 .pth로 연결 (별도 설치 없이 KORMo-tutorial/src를 import 경로에 추가)
echo "$(pwd)/KORMo-tutorial/src" > .venv/lib/python3.12/site-packages/kormo_src.pth
```

- 현재 설치: torch 2.13(MPS 지원), transformers 4.57.1, datasets 5.0, accelerate 1.14
- **flash-attn 미설치** — CUDA 전용이라 macOS에서 빌드 불가 (로컬은 sdpa 사용)

## 미니 KORMo-VL (`03_kormo_vl_mini_colab.ipynb`)

1번 노트북의 1B 모델(`output/final/`)을 백본으로, [KORMo-VL](https://huggingface.co/KORMo-VL/KORMo-VL)과
같은 LLaVA 방식으로 비전 능력을 붙이는 실습. SigLIP-base(동결) → 2층 MLP 프로젝터(**유일한 학습 대상**,
~6M) → KORMo 1B(동결) 구조로, 이미지 1장을 196개 토큰으로 변환해 텍스트 앞에 이어붙인다.

- 데이터: [KoLLaVA-Instruct-313k](https://huggingface.co/datasets/mosshoon/KoLLaVA-Instruct-313k)
  (한국어 VQA, 이미지 내장 parquet) — 샤드 단위 부분 다운로드, 기본 2샤드(~1GB, 19k쌍)
- 손실은 답변 토큰에만. A100 기준 학습 20~40분, `SMOKE=True`로 1.5k 미니셋 검증 가능
- 산출물은 Drive `kormo-1B-PT/vl/`에 저장 — LLM 백본은 읽기 전용이라 1·2번 사이클과 독립
- 검증: 조립 로직(SigLIP→프로젝터→`inputs_embeds`→KORMo, 손실 마스킹, grad 흐름, 수동 생성 루프)은
  로컬 MPS에서 1스텝 테스트 통과

## 로컬(Mac)에서 학습 실행 (`scripts/pretrain_local.py`)

Colab 노트북 파이프라인을 Apple Silicon(MPS)에서 돌 수 있게 옮긴 스크립트.
Colab 버전과의 차이: flex_attention(CUDA 전용) → **sdpa** (intra-doc mask 없음 —
패킹 시퀀스 안 문서 간 attention 허용), Drive → 로컬 `kormo-1B-PT/output/`, wandb 대신 콘솔 로깅.
3-모드 자동 분기(fresh/resume/continue)와 모드별 LR 분기는 동일.

```bash
source .venv/bin/activate
python scripts/pretrain_local.py --smoke               # 파이프라인 검증 (합성 데이터, 저장은 kormo-1B-PT-smoke/)
python scripts/pretrain_local.py --max-docs 5000       # 데이터 일부로 짧은 학습
python scripts/pretrain_local.py                       # 전체 1 epoch — M3 Pro 기준 수일 소요 주의
```

- M3 Pro 36GB 기준 1.3B bf16 학습이 메모리에는 들어가지만(가중치 2.6 + grad 2.6 + Adam ~5.2GB + 활성값),
  **속도가 A100의 수십분의 1**이라 전체 학습은 비현실적 — 파이프라인 실험·디버깅 용도
- 기본 seq 2048 / batch 1 / grad-accum 8. 메모리 부족하면 `--grad-checkpoint` 또는 `--seq-len 1024`

## 로컬(Mac)에서 가능한 작업

- 토크나이저 실험: `KORMo-Team/KORMo-tokenizer` (vocab 125,000)
- 데이터 파이프라인: 로드 → 토크나이즈 → 4096 시퀀스 패킹 → collator
- 튜토리얼 데이터셋: `KORMo-Team/KORMo-tutorial-datasets` (`name='pretrain'`)
- `pretrain_local.py`로 소규모 사전학습 실험 (위 섹션)

## GPU가 필요한 작업 (Colab A100 또는 Linux GPU 서버)

- `01.pretrain_from_scratch.ipynb` 학습 셀 — flex_attention + bf16 + CUDA 필요
- 노트북 경로가 `/content/` 기준 → Colab 상정으로 작성됨
- 1.3B 실전 학습: 단일 노드 8 GPU (`src/scripts/run_KORMo_1B_singlenode.sh`)

## 주의

- `KORMo-tutorial/src/scripts/run_KORMo_1B_singlenode.sh`에 저자 팀의 WANDB_API_KEY가 하드코딩되어 있음 — 실행 시 본인 키로 교체
- 실전 config의 `dataset_name: "hard coding"` — 데이터 경로 직접 지정 필요
- `kormo-lm` org의 일부 데이터셋은 gated (접근 신청 필요)

## 참고

- 논문: https://arxiv.org/abs/2510.09426 (KORMo-10B, 10.8B, 한국어 68.74% 합성 데이터)
- 모델/데이터: https://huggingface.co/KORMo-Team
- 튜토리얼: https://github.com/MLP-Lab/KORMo-tutorial
