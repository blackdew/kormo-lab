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
│   └── 02_kormo_eval_colab.ipynb              # 벤치마크 평가 노트북 (L4로 충분)
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
- 체크포인트는 optimizer 포함(~8GB) 최근 1개 유지, 완료 시 `final/`(2.6GB)에 배포용 저장 — Drive 무료 15GB 내
- **wandb 로깅 기본 활성화** (`USE_WANDB`) — 프로젝트 `kormo-lab`, run 이름 `kormo-1B-{모드}`.
  API 키는 Colab 보안 비밀 `WANDB_API_KEY`에서 읽음 (없으면 로그인 프롬프트)

## 평가 (`02_kormo_eval_colab.ipynb`)

Drive의 `final/` 모델을 lm-evaluation-harness(0.4.12)로 평가. L4 GPU면 충분.

1. **Held-out PPL** — 한국어 위키 200문서. 튜토리얼 스케일에서 가장 민감한 진행 지표
2. **KoBEST + HAE-RAE** — loglikelihood 채점이라 작은 base 모델도 측정 가능 (0-shot, ~10분)
3. **KMMLU** — `RUN_KMMLU=True`로 활성화 (5-shot, KORMo 논문 설정, 1시간+)

결과는 `MyDrive/kormo-1B-PT/evals/eval_*.json`에 누적 — 추가 학습 반복 시 성능 곡선 비교용

## 환경

```bash
source .venv/bin/activate
```

- 의존성은 PyPI에서 직접 설치됨 (transformers 4.57.1, torch, datasets, accelerate, trl 등)
- `kormo` 패키지는 `.pth` 파일로 `KORMo-tutorial/src` 경로 연결 (별도 설치 없음)
- **flash-attn 미설치** — CUDA 전용이라 macOS에서 빌드 불가

## 로컬(Mac)에서 가능한 작업

- 토크나이저 실험: `KORMo-Team/KORMo-tokenizer` (vocab 125,000)
- 데이터 파이프라인: 로드 → 토크나이즈 → 4096 시퀀스 패킹 → collator
- 튜토리얼 데이터셋: `KORMo-Team/KORMo-tutorial-datasets` (`name='pretrain'`)

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
