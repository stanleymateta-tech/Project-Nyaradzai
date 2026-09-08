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
    parser.add_argument("--batch",      type=int, default=2)
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.config.use_cache = False

    print("Loading WAXAL data...")
    import soundfile as sf_lib
    import io as io_lib

    ds_raw = load_dataset(
        "google/WaxalNLP", "sna_asr",
        split="train",
    )

    N = min(args.max_steps * args.batch, 2000)
    print(f"Processing {N} examples...")

    processed = []
    for i in range(min(N, len(ds_raw))):
        try:
            example = ds_raw[i]
            # Get audio bytes and decode with soundfile
            audio_data = example.get("audio", {})
            if isinstance(audio_data, dict) and "bytes" in audio_data:
                arr, sr = sf_lib.read(io_lib.BytesIO(audio_data["bytes"]))
            elif isinstance(audio_data, dict) and "array" in audio_data:
                arr = audio_data["array"]
                sr  = audio_data.get("sampling_rate", 16000)
            else:
                continue
            import numpy as np_inner
            if len(arr.shape) > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa as lb
                arr = lb.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)
            text = example.get("transcription") or example.get("text","")
            if not isinstance(text, str) or len(text) < 2:
                continue
            feat   = processor.feature_extractor(
                arr.astype(np.float32), sampling_rate=16000,
                return_tensors="pt").input_features[0]
            labels = processor.tokenizer(text).input_ids
            processed.append({"input_features": feat.numpy(), "labels": labels})
        except Exception:
            continue
        if (i+1) % 200 == 0:
            print(f"  Processed {len(processed)} valid examples so far...")

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
        gradient_accumulation_steps = 8,
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
