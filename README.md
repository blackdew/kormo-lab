# kormo-lab

KORMo(Korean Open Reasoning Model) 기반 한국어 LLM from-scratch 학습 실습 프로젝트.

## 구성

```
kormo-lab/
├── KORMo-tutorial/   # MLP-Lab/KORMo-tutorial 클론 (외부 코드, git 추적 제외)
│   ├── tutorial/     # 01.pretrain_from_scratch → 02.sft_qlora → 03.inference
│   ├── src/kormo/    # KORMo 아키텍처 구현 + KORMoTrainer
│   └── src/scripts/  # 실전용 accelerate 실행 스크립트 (1B/10B)
├── .venv/            # Python 3.12 (uv), git 추적 제외
└── README.md
```

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
