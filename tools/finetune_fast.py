#!/usr/bin/env python3
"""
finetune_fast.py — Fast Whisper Shona fine-tuning
Skips slow preprocessing by using streaming mode and smaller batches.
Works on Kaggle without getting stuck.
"""
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Starsm91/whisper-small-shona")
    parser.add_argument("--output",     default="/kaggle/working/out")
    parser.add_argument("--max-steps",  type=int, default=1000)
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--hub-repo",   default="Starsm91/whisper-small-shona")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN","")

    print("Installing/checking packages...")
    import subprocess
    subprocess.run([sys.executable,"-m","pip","install","-q",
        "transformers","torch","datasets","accelerate",
        "evaluate","jiwer","soundfile"], check=True)

    import torch
    from datasets import load_dataset, Audio, IterableDataset
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer
    )
    import evaluate
    from dataclasses import dataclass
    from typing import Any, List
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model: {args.model}")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="shona", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "shona"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print("Loading WAXAL data in streaming mode (no preprocessing wait)...")
    # Use streaming=True to avoid the slow Map step
    ds_stream = load_dataset(
        "google/WaxalNLP", "sna_asr",
        split="train",
        streaming=True,
        trust_remote_code=True
    ).cast_column("audio", Audio(sampling_rate=16000))

    # Convert stream to list of processed examples (first N only)
    N = min(args.max_steps * args.batch, 8000)
    print(f"Processing {N} examples on the fly...")

    processed = []
    for i, example in enumerate(ds_stream):
        if i >= N:
            break
        try:
            audio = example["audio"]["array"]
            sr    = example["audio"]["sampling_rate"]
            text  = example.get("transcription") or example.get("text","")
            if not text or len(text) < 2:
                continue
            feat   = processor.feature_extractor(
                audio, sampling_rate=sr, return_tensors="pt").input_features[0]
            labels = processor.tokenizer(text).input_ids
            processed.append({"input_features": feat.numpy(), "labels": labels})
        except Exception:
            continue
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{N} examples")

    print(f"Ready: {len(processed)} examples")

    from datasets import Dataset
    import numpy as np

    # Build a simple dataset
    train_ds = Dataset.from_dict({
        "input_features": [p["input_features"] for p in processed],
        "labels":         [p["labels"]          for p in processed],
    })

    @dataclass
    class DataCollator:
        processor: Any
        def __call__(self, features: List[dict]) -> dict:
            inputs = torch.tensor(
                np.array([f["input_features"] for f in features]),
                dtype=torch.float32)
            labels = [f["labels"] for f in features]
            max_len = max(len(l) for l in labels)
            padded  = [l+[-100]*(max_len-len(l)) for l in labels]
            return {"input_features": inputs,
                    "labels": torch.tensor(padded, dtype=torch.long)}

    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids==-100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        print(f"\n  WER: {wer*100:.2f}%\n")
        return {"wer": wer}

    os.makedirs(args.output, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args.output,
        per_device_train_batch_size = args.batch,
        gradient_accumulation_steps = 2,
        learning_rate               = 1e-5,
        warmup_steps                = 100,
        max_steps                   = args.max_steps,
        fp16                        = torch.cuda.is_available(),
        eval_strategy               = "steps",
        eval_steps                  = 200,
        save_steps                  = 200,
        logging_steps               = 25,
        predict_with_generate       = True,
        generation_max_length       = 225,
        push_to_hub                 = bool(token),
        hub_model_id                = args.hub_repo,
        hub_strategy                = "checkpoint",
        hub_token                   = token,
        report_to                   = ["none"],
    )

    eval_size = min(200, len(train_ds)//10)
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = train_ds.select(range(eval_size)),
        data_collator   = DataCollator(processor),
        compute_metrics = compute_metrics,
        processing_class= processor.feature_extractor,
    )

    print(f"\nTraining for {args.max_steps} steps...")
    print("Model saves to Hugging Face every 200 steps automatically.\n")
    trainer.train()

    if token:
        print("Publishing final model...")
        trainer.push_to_hub()
        processor.push_to_hub(args.hub_repo, token=token)
        print(f"PUBLISHED: https://huggingface.co/{args.hub_repo}")
    else:
        print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
