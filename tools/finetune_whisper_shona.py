#!/usr/bin/env python3
"""
finetune_whisper_shona.py — Fine-tune OpenAI Whisper on Google's WAXAL
Shona ASR dataset (huggingface.co/datasets/google/WaxalNLP, config sna_asr).

This turns Phase 3 (Shona speech recognition for TV/video captioning) from
a data-collection project into a training run. WAXAL is CC-BY-4.0, so the
resulting model can be released openly (credit Google's WAXAL in any release).

WHERE TO RUN (free options):
  - Google Colab (free T4 GPU): whisper-small fits comfortably
  - Kaggle notebooks (free GPU quota)
  Upload this file, run: !pip install -q transformers datasets evaluate
                                       jiwer accelerate soundfile librosa
  then: !python finetune_whisper_shona.py --model openai/whisper-small

OUTPUT: a Shona-specialised Whisper checkpoint + word-error-rate report,
publishable to Hugging Face under your own name/organisation.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small",
                    help="whisper-tiny for quick tests, whisper-small for Colab, "
                         "whisper-large-v3 if you have a big GPU")
    ap.add_argument("--output", default="./whisper-shona")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    import torch
    import evaluate
    from dataclasses import dataclass
    from datasets import load_dataset, Audio
    from transformers import (WhisperProcessor,
                              WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    print("Loading WAXAL Shona ASR data (google/WaxalNLP, sna_asr)...")
    ds = load_dataset("google/WaxalNLP", "sna_asr")
    # Whisper expects 16 kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # find the transcript column (dataset cards evolve; be defensive)
    text_col = next(c for c in ds["train"].column_names
                    if c.lower() in ("text", "transcription", "transcript", "sentence"))
    print(f"Transcript column: {text_col};  train size: {len(ds['train']):,}")

    processor = WhisperProcessor.from_pretrained(
        args.model, language="shona", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "shona"
    model.generation_config.task = "transcribe"

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    try:
        ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1)
    except RuntimeError as e:
        print(f"Map failed ({e}). Retrying with batched=False...")
        ds = ds.map(prepare, remove_columns=ds["train"].column_names,
                    num_proc=1, batched=False)

    @dataclass
    class Collator:
        def __call__(self, features):
            input_feats = [{"input_features": f["input_features"]} for f in features]
            batch = processor.feature_extractor.pad(input_feats, return_tensors="pt")
            label_feats = [{"input_ids": f["labels"]} for f in features]
            labels = processor.tokenizer.pad(label_feats, return_tensors="pt")
            lab = labels["input_ids"].masked_fill(
                labels.attention_mask.ne(1), -100)
            if (lab[:, 0] == processor.tokenizer.bos_token_id).all():
                lab = lab[:, 1:]
            batch["labels"] = lab
            return batch

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        ref = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=hyp, references=ref)}

    eval_split = "validation" if "validation" in ds else (
        "test" if "test" in ds else None)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=min(args.batch, 4),
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=200,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps" if eval_split else "no",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds[eval_split] if eval_split else None,
        data_collator=Collator(),
        compute_metrics=compute_metrics if eval_split else None,
        processing_class=processor.feature_extractor,
    )
    import glob
    ckpts = glob.glob(f"{args.output}/checkpoint-*")
    if ckpts:
        print(f"Found {len(ckpts)} checkpoint(s) — resuming training.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"\nDone. Shona Whisper model saved to {args.output}")
    print("Next: push to Hugging Face hub, then wire into a captioning "
          "pipeline (whisper-live / whisperX) for broadcast use.")


if __name__ == "__main__":
    main()
