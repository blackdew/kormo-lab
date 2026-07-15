#!/usr/bin/env python
"""KORMo 1B 사전학습 — 로컬 macOS(Apple Silicon/MPS) 버전.

notebooks/01_kormo_1B_pretrain_colab_a100.ipynb의 파이프라인을 Mac에서 돌 수 있게 옮긴 스크립트.
Colab(A100) 버전과의 차이:
  - flex_attention(CUDA 전용) → sdpa. intra-document mask가 없어져 패킹된 시퀀스 안에서
    문서 간 attention이 허용됨 (GPT-2 시절 표준 방식 — 튜토리얼 스케일에서 영향 미미)
  - Google Drive → --base-dir (기본 ./kormo-1B-PT) 아래 output/ 에 저장
모드 자동 분기(fresh/resume/continue)와 학습률 분기는 노트북과 동일.
wandb 로깅도 노트북과 동일하게 동작: 프로젝트 kormo-lab, resume 모드는 저장된 run ID로
이전 run에 이어 기록, 종료 시 wandb.finish()로 Finished 마감.
(--no-wandb로 끔. smoke 모드와 wandb 미설치·미로그인 환경은 자동으로 콘솔 로깅)

사용 예:
  python scripts/pretrain_local.py --smoke               # 파이프라인 동작 검증 (합성 데이터 2스텝)
  python scripts/pretrain_local.py --max-docs 5000       # 데이터 일부로 짧은 학습
  python scripts/pretrain_local.py                       # 전체 데이터 1 epoch (수일 소요 주의)
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import datetime

# torch/huggingface_hub import 전에 설정해야 적용됨
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # MPS 미지원 연산은 CPU 폴백
if not os.environ.get('HF_TOKEN'):
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')        # 익명 Xet 접근은 401이 나므로


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir', default='kormo-1B-PT', help='저장 루트 (output/ 이 아래 생김)')
    p.add_argument('--seq-len', type=int, default=2048, help='패킹 시퀀스 길이 (Colab은 4096, 로컬 기본 2048)')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8, help='유효 배치 = batch-size × grad-accum')
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--max-steps', type=int, default=-1, help='지정 시 epochs 무시하고 이 스텝 수만 학습')
    p.add_argument('--max-docs', type=int, default=0, help='0이면 전체, 아니면 앞에서 N개 문서만 사용')
    p.add_argument('--lr', type=float, default=0.0, help='0이면 모드별 자동 (fresh/resume 5e-4, continue 5e-5)')
    p.add_argument('--val-size', type=int, default=64, help='검증 시퀀스 수 (0이면 평가 생략)')
    p.add_argument('--eval-steps', type=int, default=200)
    p.add_argument('--save-steps', type=int, default=200)
    p.add_argument('--grad-checkpoint', action='store_true', help='gradient checkpointing (메모리↓ 속도↓)')
    p.add_argument('--force-fresh', action='store_true', help='기존 output/을 백업하고 처음부터')
    p.add_argument('--smoke', action='store_true', help='합성 데이터로 파이프라인 검증 (다운로드 최소화)')
    p.add_argument('--no-wandb', action='store_true', help='wandb 로깅 끄기 (smoke 모드는 자동 off)')
    return p.parse_args()


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return 'mps'
    print('경고: MPS 사용 불가 — CPU로 학습합니다 (매우 느림)')
    return 'cpu'


def resolve_mode(base_dir, force_fresh):
    """노트북과 동일한 3-모드 자동 분기 + FORCE_FRESH 백업."""
    output_dir = os.path.join(base_dir, 'output')
    if force_fresh and os.path.isdir(output_dir):
        backup = os.path.join(base_dir, f'output-backup-{datetime.now():%Y%m%d-%H%M%S}')
        shutil.move(output_dir, backup)
        print(f'기존 결과 백업됨: {backup}')

    final_dir = os.path.join(output_dir, 'final')
    has_ckpt = bool(glob.glob(os.path.join(output_dir, 'checkpoint-*')))
    if os.path.isdir(final_dir):
        mode = 'continue'
    elif has_ckpt:
        mode = 'resume'
    else:
        mode = 'fresh'
    return mode, output_dir, final_dir


def setup_wandb(args, mode, output_dir):
    """노트북과 동일한 wandb run 관리 — resume이면 저장된 run ID로 이어 기록.

    run의 정체성은 이름이 아니라 ID: resume 모드는 output/에 저장해둔 ID를 재사용해
    끊긴 run의 곡선에 이어붙이고, 그 외 모드는 global_step이 0부터 시작하므로
    새 run을 발급한다 (같은 run에 낮은 스텝을 다시 쓰면 wandb가 로그를 거부함).
    반환: (report_to, run_name)
    """
    if args.no_wandb or args.smoke:
        return 'none', None
    try:
        import wandb  # noqa: F401
    except ImportError:
        print('wandb 미설치 — 콘솔 로깅으로 폴백 (uv pip install wandb)')
        return 'none', None

    os.environ.setdefault('WANDB_PROJECT', 'kormo-lab')
    run_name = f'kormo-1B-local-{mode}'
    run_id_file = os.path.join(output_dir, 'wandb_run_id.txt')
    if mode == 'resume' and os.path.exists(run_id_file):
        with open(run_id_file) as f:
            run_id = f.read().strip()
        print(f'wandb: 이전 세션 run에 이어서 기록 — {run_id}')
    else:
        run_id = f'{run_name}-{datetime.now():%Y%m%d-%H%M%S}'
        os.makedirs(output_dir, exist_ok=True)
        with open(run_id_file, 'w') as f:
            f.write(run_id)
    os.environ['WANDB_RUN_ID'] = run_id
    os.environ['WANDB_RESUME'] = 'allow'   # 같은 ID의 run이 있으면 이어서, 없으면 새로 생성
    return 'wandb', run_name


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('KORMo-Team/KORMo-tokenizer')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(args, tokenizer):
    """Raw text → tokenize → seq-len 패킹 → train/val 분리 (노트북 파이프라인과 동일 seed)."""
    import datasets
    from itertools import chain

    num_proc = os.cpu_count()

    if args.smoke:
        # 합성 데이터: 다운로드 없이 학습 루프만 검증
        import torch
        n_seq, vocab = 24, tokenizer.vocab_size
        data = torch.randint(3, vocab, (n_seq, args.seq_len)).tolist()
        packed = datasets.Dataset.from_dict({'input_ids': data})
    else:
        raw = datasets.load_dataset('KORMo-Team/KORMo-tutorial-datasets', name='pretrain', split='train')
        raw = raw.shuffle(seed=42)
        if args.max_docs:
            raw = raw.select(range(min(args.max_docs, len(raw))))

        def _tokenize(examples):
            ids = [tokenizer.encode(t) + [tokenizer.eos_token_id] for t in examples['text']]
            return {'input_ids': ids}

        tokenized = raw.map(_tokenize, batched=True, num_proc=num_proc, remove_columns=raw.column_names)

        def _pack(examples):
            flat = list(chain.from_iterable(examples['input_ids']))
            n = len(flat) // args.seq_len
            return {'input_ids': [flat[i * args.seq_len:(i + 1) * args.seq_len] for i in range(n)]}

        packed = tokenized.map(_pack, batched=True, batch_size=100_000,
                               num_proc=num_proc, remove_columns=tokenized.column_names)

    packed.set_format('torch')
    val_size = min(args.val_size, max(len(packed) - 2, 0))
    if val_size > 0:
        split = packed.train_test_split(test_size=val_size, seed=42)
        return split['train'], split['test']
    return packed, None


def build_model(mode, final_dir, device):
    """1B config로 초기화(fresh/resume) 또는 final/ 로드(continue). attention은 sdpa."""
    import torch
    import yaml
    from importlib.resources import files
    from kormo.model._configuration_kormo import KORMoConfig
    from kormo.model._modeling_kormo import KORMoForCausalLM

    if mode == 'continue':
        model = KORMoForCausalLM.from_pretrained(
            final_dir, dtype=torch.bfloat16, _attn_implementation='sdpa')
    else:
        cfg_text = (files('kormo.modeling_configs') / 'kormo_1B.yaml').read_text()
        cfg_dict = yaml.safe_load(cfg_text)
        cfg_dict['_attn_implementation'] = 'sdpa'   # flash_attention_2(CUDA 전용) 대체
        model = KORMoForCausalLM._from_config(KORMoConfig(**cfg_dict))  # yaml의 dtype=bfloat16 적용됨

    return model.to(device)


def main():
    args = parse_args()
    if args.smoke:
        # smoke의 랜덤 가중치 final/이 실제 학습 폴더에 남으면 다음 실행이
        # continue 모드로 잘못 시작하므로 저장 위치를 분리
        args.base_dir = args.base_dir.rstrip('/') + '-smoke'
    device = pick_device()
    mode, output_dir, final_dir = resolve_mode(args.base_dir, args.force_fresh)
    lr = args.lr or (5e-5 if mode == 'continue' else 5e-4)
    report_to, run_name = setup_wandb(args, mode, output_dir)
    print(f'device: {device} | 모드: {mode} | LR: {lr} | 저장: {output_dir} | 로깅: {report_to}')

    import torch
    from dataclasses import dataclass
    from collections import defaultdict
    from torch.nn.utils.rnn import pad_sequence
    from transformers import PreTrainedTokenizer
    from kormo.train.arguments import KORMoTrainingArguments
    from kormo.train.trainer import KORMoTrainer

    tokenizer = load_tokenizer()
    train_ds, val_ds = build_dataset(args, tokenizer)
    total_tokens = len(train_ds) * args.seq_len
    print(f'학습: {len(train_ds)} 시퀀스 ({total_tokens / 1e6:.1f}M 토큰) | '
          f'검증: {len(val_ds) if val_ds is not None else 0} 시퀀스')

    model = build_model(mode, final_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{mode}] 파라미터: {n_params / 1e9:.2f}B | attn: {model.config._attn_implementation}'
          f' | dtype: {next(model.parameters()).dtype}')

    @dataclass
    class CausalCollator:
        """sdpa용 collator — 노트북의 flex_attention block mask 대신 표준 causal mask 사용."""
        tokenizer: PreTrainedTokenizer

        def __call__(self, instances):
            input_ids = pad_sequence([inst['input_ids'][:args.seq_len] for inst in instances],
                                     batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return dict(input_ids=input_ids, labels=labels,
                        attention_mask=torch.ones_like(input_ids))

    do_eval = val_ds is not None
    training_arguments = KORMoTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_checkpoint,
        learning_rate=lr,
        lr_scheduler_type='linear',
        logging_steps=5,
        eval_strategy='steps' if do_eval else 'no',
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=1,
        save_only_model=False,
        report_to=report_to,
        run_name=run_name,
        dataloader_pin_memory=False,   # MPS에서는 pin_memory 미지원 경고 방지
    )

    trainer = KORMoTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=CausalCollator(tokenizer),
    )
    # 업스트림 버그 우회: _metrics에 'eval' 키가 없어 첫 평가에서 KeyError (노트북과 동일)
    trainer._metrics.setdefault('eval', defaultdict(list))

    if mode == 'resume':
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print('최종 모델 저장:', final_dir)

    model.eval()
    inputs = tokenizer('한국의 수도는', return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=True, top_p=0.9)
    print('생성 테스트:', tokenizer.decode(out[0], skip_special_tokens=True))

    # run을 Finished 상태로 마감 — 없으면 프로세스 강제 종료 시 Crashed로 표기
    if report_to == 'wandb':
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
