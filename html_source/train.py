"""
SageMaker HuggingFace training script for HTML/DOM phishing classification.
Fine-tunes distilbert-base-uncased (binary: 0 = benign, 1 = phishing).
"""

import argparse
import json
import os

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name_or_path',          type=str,   default='distilbert-base-uncased')
    p.add_argument('--num_labels',                  type=int,   default=2)
    p.add_argument('--num_train_epochs',            type=int,   default=3)
    p.add_argument('--per_device_train_batch_size', type=int,   default=32)
    p.add_argument('--per_device_eval_batch_size',  type=int,   default=64)
    p.add_argument('--learning_rate',               type=float, default=2e-5)
    p.add_argument('--warmup_steps',                type=int,   default=200)
    p.add_argument('--weight_decay',                type=float, default=0.01)
    p.add_argument('--max_seq_length',              type=int,   default=128)
    p.add_argument('--output_dir',                  type=str,   default='/opt/ml/model')

    # SageMaker channel dirs (injected via env vars)
    p.add_argument(
        '--training_dir',
        type=str,
        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'),
    )
    p.add_argument(
        '--validation_dir',
        type=str,
        default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_channel(channel_dir: str) -> pd.DataFrame:
    """Concatenate all CSVs in a SageMaker input channel directory."""
    frames = []
    for fname in sorted(os.listdir(channel_dir)):
        if fname.endswith('.csv'):
            frames.append(pd.read_csv(os.path.join(channel_dir, fname)))
    if not frames:
        raise FileNotFoundError(f'No CSV files found in {channel_dir}')
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f'[data]  train      = {args.training_dir}')
    print(f'[data]  validation = {args.validation_dir}')

    train_df = load_csv_channel(args.training_dir)
    val_df   = load_csv_channel(args.validation_dir)

    for df, split in [(train_df, 'train'), (val_df, 'validation')]:
        assert {'label', 'text'}.issubset(df.columns), \
            f'{split} CSV must contain "label" and "text" columns (got: {list(df.columns)})'
        df['label'] = df['label'].astype(int)
        df['text']  = df['text'].astype(str).fillna('')

    print(f'[data]  train samples      = {len(train_df)}')
    print(f'[data]  validation samples = {len(val_df)}')
    print('[data]  label distribution (train):\n',
          train_df['label'].value_counts().to_string())

    # ── Tokenise ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize(batch):
        return tokenizer(
            batch['text'],
            max_length=args.max_seq_length,
            truncation=True,
            padding='max_length',
        )

    train_ds = Dataset.from_pandas(train_df[['label', 'text']].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[['label', 'text']].reset_index(drop=True))

    train_ds = train_ds.map(tokenize, batched=True, batch_size=512, remove_columns=['text'])
    val_ds   = val_ds.map(tokenize,   batched=True, batch_size=512, remove_columns=['text'])

    train_ds.set_format('torch')
    val_ds.set_format('torch')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy_metric = evaluate.load('accuracy')
    f1_metric       = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
        f1  = f1_metric.compute(predictions=preds,  references=labels, average='binary')['f1']
        return {'accuracy': acc, 'f1': f1}

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.num_train_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        learning_rate               = args.learning_rate,
        warmup_steps                = args.warmup_steps,
        weight_decay                = args.weight_decay,
        evaluation_strategy         = 'epoch',
        save_strategy               = 'epoch',
        load_best_model_at_end      = True,
        metric_for_best_model       = 'f1',
        greater_is_better           = True,
        logging_steps               = 50,
        report_to                   = 'none',
        fp16                        = True,   # Training job runs on G5 (NVIDIA A10G, Tensor Cores)
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print('[train] starting fine-tuning ...')
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    # safe_serialization=False → writes pytorch_model.bin instead of model.safetensors.
    # Required: the HuggingFace SageMaker inference DLC expects pytorch_model.bin.
    trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    print(f'[train] model saved to {args.output_dir}')

    # Persist final eval metrics next to the model
    metrics = trainer.evaluate()
    print('[eval]  final metrics:', metrics)
    with open(os.path.join(args.output_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == '__main__':
    main()
