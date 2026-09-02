#!/usr/bin/env python3
"""
synthetic_training_data.py — Generate synthetic Shona training data using TTS
Project Nyaradzai / Rurimi RwaAmai

Uses Meta MMS-TTS to convert Shona text into audio, then pairs
audio + text as training data for the ASR model.
This lets books, novels, Bible passages, and news articles
become training data WITHOUT needing human recordings.

RUN ON KAGGLE (GPU recommended):
    pip install transformers torch soundfile datasets

PROCESS:
    1. Reads Shona text from a file (one sentence per line)
    2. Generates audio for each sentence using MMS-TTS
    3. Saves audio + text pairs as a Hugging Face dataset
    4. Upload to Starsm91/shona-synthetic-corpus on Hugging Face
    5. Use alongside WAXAL when retraining the ASR model
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

def generate_synthetic_corpus(text_file, output_dir="./synthetic_corpus",
                               max_sentences=500):
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline

    Path(output_dir).mkdir(exist_ok=True)
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("Loading Meta MMS-TTS Shona voice...")
    tts = pipeline("text-to-speech", model="facebook/mms-tts-sna")
    print("TTS model ready.\n")

    # Read sentences
    if text_file and Path(text_file).exists():
        sentences = [l.strip() for l in Path(text_file).read_text().splitlines()
                     if l.strip() and len(l.strip()) > 10]
    else:
        # Built-in sample sentences for testing
        sentences = [
            "Mangwanani akanaka, makadii here?",
            "Ndinotenda chaizvo nerubatsiro rwenyu.",
            "Zimbabwe nyika yedu yakanaka uye ine vanhu vakanaka.",
            "Rurimi rwedu nderwedu tose, tinofanira kurichengetedza.",
            "Vana vedu ngavadzidze kutaura Shona neChirungu zvakanaka.",
            "Mwari akasika vanhu vose vakafanana.",
            "Tiri pamwechete sevanhu veZimbabwe, tine simba.",
            "Ndinoda kudzidza zvakawanda nezve nyika yangu.",
            "Musha mwoyo wangu uri kuZimbabwe.",
            "Tine tariro yakanaka yemangwana edu.",
        ]

    sentences = sentences[:max_sentences]
    print(f"Processing {len(sentences)} sentences...")

    metadata = []
    for i, text in enumerate(sentences):
        try:
            result = tts(text)
            audio = np.array(result["audio"]).squeeze()
            sr    = result["sampling_rate"]
            fname = f"syn_{i:05d}.wav"
            sf.write(audio_dir / fname, audio, sr)
            metadata.append({
                "audio_path": f"audio/{fname}",
                "transcription": text,
                "source": "mms-tts-synthetic",
                "sampling_rate": sr
            })
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(sentences)} done")
        except Exception as e:
            print(f"  Skipped sentence {i}: {e}")

    # Save metadata
    meta_path = Path(output_dir) / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\nDone! Generated {len(metadata)} synthetic training pairs")
    print(f"Saved to: {output_dir}")
    print("\nNEXT: Upload to Hugging Face as Starsm91/shona-synthetic-corpus")
    print("Then add to the Kaggle training notebook alongside WAXAL data")
    return metadata

if __name__ == "__main__":
    text_file = sys.argv[1] if len(sys.argv) > 1 else None
    max_sent  = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    generate_synthetic_corpus(text_file, max_sentences=max_sent)
