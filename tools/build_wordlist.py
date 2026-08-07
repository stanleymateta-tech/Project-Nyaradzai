#!/usr/bin/env python3
"""
build_wordlist.py — Grow the Shona dictionary from real text.

Feed it any Shona text (news articles from Kwayedza, the Shona Bible,
Shona Wikipedia dump, school textbooks, JW300 corpus, radio transcripts)
and it extracts a frequency-ranked wordlist, filters noise, and merges
new words into the Hunspell dictionary.

Usage:
    python3 build_wordlist.py corpus1.txt corpus2.txt --min-count 3 \
        --dic ../dictionaries/sn_ZW.dic --out merged.dic

The more text you feed it, the better the spellchecker gets.
This is the tool to hand to university linguistics departments:
students can each contribute cleaned text, and the dictionary compounds.
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Shona orthography: a-z plus apostrophe in n' (velar nasal: n'anga, imbwa yavo)
TOKEN_RE = re.compile(r"[a-z](?:[a-z']*[a-z])?", re.IGNORECASE)

# Letters that never occur in standard Shona orthography (flags English/other)
NON_SHONA_LETTERS = set("qxl")  # 'l' only in loanwords/names

# Common English words to exclude when corpora are code-mixed
ENGLISH_STOP = set("""the and of to in is that for on with as are was be this
it at by an or from not but have has had they you we he she his her their
""".split())


def tokenize(text: str):
    text = unicodedata.normalize("NFC", text)
    for m in TOKEN_RE.finditer(text):
        yield m.group(0).lower()


def looks_shona(word: str) -> bool:
    if len(word) < 2 or len(word) > 30:
        return False
    if word in ENGLISH_STOP:
        return False
    if any(c in NON_SHONA_LETTERS for c in word):
        return False
    # Shona syllables end in vowels; a word ending in a consonant
    # (other than n' contexts) is almost always foreign
    if word[-1] not in "aeiou":
        return False
    return True


def load_dic(path: Path) -> set:
    words = set()
    if not path or not path.exists():
        return words
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if i == 0 or line.startswith("#") or not line.strip():
            continue
        words.add(line.split("/")[0].strip().lower())
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", help="Plain-text Shona corpus files")
    ap.add_argument("--min-count", type=int, default=3,
                    help="Minimum frequency to accept a word (default 3)")
    ap.add_argument("--dic", type=Path, default=None,
                    help="Existing sn_ZW.dic to merge with")
    ap.add_argument("--out", type=Path, default=Path("merged.dic"))
    ap.add_argument("--freq-out", type=Path, default=Path("frequencies.tsv"),
                    help="Frequency table (used by autocorrect ranking)")
    args = ap.parse_args()

    counts = Counter()
    for fp in args.corpora:
        text = Path(fp).read_text(encoding="utf-8", errors="replace")
        counts.update(w for w in tokenize(text) if looks_shona(w))
        print(f"  processed {fp}: {sum(counts.values()):,} tokens so far",
              file=sys.stderr)

    existing = load_dic(args.dic)
    new_words = sorted(w for w, c in counts.items()
                       if c >= args.min_count and w not in existing)

    all_words = sorted(existing | set(new_words))
    with args.out.open("w", encoding="utf-8") as f:
        f.write(f"{len(all_words)}\n")
        for w in all_words:
            f.write(w + "\n")

    with args.freq_out.open("w", encoding="utf-8") as f:
        for w, c in counts.most_common():
            f.write(f"{w}\t{c}\n")

    print(f"\n{len(new_words):,} new words found (freq >= {args.min_count})")
    print(f"{len(all_words):,} total -> {args.out}")
    print(f"Frequency table -> {args.freq_out}")
    print("\nNOTE: review new words before release — corpus text contains "
          "typos too. Sort by frequency and have a Shona speaker approve.")


if __name__ == "__main__":
    main()
