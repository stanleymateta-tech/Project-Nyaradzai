#!/usr/bin/env python3
"""
waxal_to_corpus.py — Extract the Shona transcripts from Google's WAXAL
dataset as a plain-text corpus, ready for build_wordlist.py.

WAXAL's ASR transcripts are real, natural, spoken Shona from diverse
speakers — exactly the vocabulary a spellchecker and phone keyboard need
(everyday speech, not just formal news/Bible register). One dataset,
two uses: audio trains the speech model, text grows the lexicon.

Run anywhere with internet (Colab is fine, no GPU needed):
    pip install -q datasets
    python3 waxal_to_corpus.py --out waxal_shona.txt
    python3 build_wordlist.py waxal_shona.txt --min-count 2 \
        --dic ../dictionaries/sn_ZW.dic --out grown.dic
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="waxal_shona.txt")
    args = ap.parse_args()

    from datasets import load_dataset
    # stream=True avoids downloading the audio — we only want text
    ds = load_dataset("google/WaxalNLP", "sna_asr", streaming=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for split in ds:
            for row in ds[split]:
                text = None
                for key in ("text", "transcription", "transcript", "sentence"):
                    if key in row and row[key]:
                        text = row[key]
                        break
                if text:
                    f.write(text.strip() + "\n")
                    n += 1
                    if n % 1000 == 0:
                        print(f"  {n:,} transcripts...")
    print(f"\n{n:,} Shona transcripts -> {args.out}")
    print("Now run build_wordlist.py on it to grow the dictionary "
          "with real spoken-Shona vocabulary and frequencies.")


if __name__ == "__main__":
    main()
