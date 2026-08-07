#!/usr/bin/env python3
"""
autocorrect.py — Shona autocorrect engine (prototype).

This is the logic that would sit inside a phone keyboard or word processor:
given a typed word, decide if it is Shona, and if not, propose the most
likely intended Shona word — WITHOUT "correcting" valid Shona into English
or Indonesian, which is the problem with today's autocorrect systems.

It expands the Hunspell dictionary into surface forms (stems x prefixes),
then ranks candidates by weighted Damerau-Levenshtein edit distance,
preferring (a) small edits, (b) frequent words, (c) phonetically close
substitutions common in Shona typing (l<->r, missing h in bh/dh/vh...).

Usage:
    python3 autocorrect.py --dic ../dictionaries/sn_ZW.dic \
        --aff ../dictionaries/sn_ZW.aff "ndinotendaa zvakanka mhorooi"
"""
import argparse
import re
import sys
from pathlib import Path

# Substitution costs tuned for Shona: these pairs are cheap because
# they are common typing/orthography confusions
CHEAP_SUBS = {("l", "r"), ("r", "l"), ("b", "bh"), ("d", "dh"), ("v", "vh"),
              ("e", "i"), ("o", "u")}


def parse_aff_prefixes(aff_path: Path):
    """Extract prefix strings per flag from the .aff file."""
    prefixes = {}
    for line in aff_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "PFX" and parts[2] == "0":
            prefixes.setdefault(parts[1], []).append(parts[3])
    suffixes = {}
    for line in aff_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "SFX" and parts[2] != "Y":
            # SFX P a i a  -> strip 'a', add 'i'
            suffixes.setdefault(parts[1], []).append((parts[2], parts[3]))
    return prefixes, suffixes


def expand(dic_path: Path, aff_path: Path):
    """Generate all surface forms the spellchecker accepts."""
    prefixes, suffixes = parse_aff_prefixes(aff_path)
    words = set()
    lines = dic_path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, flags = line.partition("/")
        words.add(word)
        stems = [word]
        for flag, subs in suffixes.items():
            if flag in flags:
                for strip, add in subs:
                    if word.endswith(strip):
                        stems.append(word[: -len(strip)] + add)
        for flag, pfx_list in prefixes.items():
            if flag in flags:
                for p in pfx_list:
                    for s in stems:
                        words.add(p + s)
        words.update(stems)
    return words


def dl_distance(a: str, b: str, max_d: int = 3) -> float:
    """Damerau-Levenshtein with cheap Shona-typical substitutions."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev2, prev, cur = None, list(range(len(b) + 1)), None
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            sub_cost = 0 if ca == cb else (
                0.4 if (ca, cb) in CHEAP_SUBS else 1)
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub_cost)
            if prev2 and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 0.7)  # transposition
        prev2, prev = prev, cur
    return prev[-1]


class ShonaAutocorrect:
    def __init__(self, dic: Path, aff: Path, freq: Path | None = None):
        self.words = expand(dic, aff)
        self.freq = {}
        if freq and freq.exists():
            for line in freq.read_text(encoding="utf-8").splitlines():
                w, _, c = line.partition("\t")
                self.freq[w] = int(c or 1)
        # bucket by length for speed
        self.by_len = {}
        for w in self.words:
            self.by_len.setdefault(len(w), []).append(w)

    def check(self, word: str) -> bool:
        return word.lower() in self.words

    def suggest(self, word: str, n: int = 5):
        w = word.lower()
        if w in self.words:
            return []
        cands = []
        for L in range(max(1, len(w) - 2), len(w) + 3):
            for c in self.by_len.get(L, []):
                d = dl_distance(w, c)
                if d <= 2.2:
                    score = d - 0.000001 * self.freq.get(c, 1)
                    cands.append((score, c))
        cands.sort()
        return [c for _, c in cands[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dic", type=Path, required=True)
    ap.add_argument("--aff", type=Path, required=True)
    ap.add_argument("--freq", type=Path, default=None)
    ap.add_argument("text", help="Text to check (quoted)")
    args = ap.parse_args()

    ac = ShonaAutocorrect(args.dic, args.aff, args.freq)
    print(f"Loaded {len(ac.words):,} surface forms\n")
    for token in re.findall(r"[a-zA-Z']+", args.text):
        if ac.check(token):
            print(f"  {token:<20} OK")
        else:
            sugg = ac.suggest(token)
            print(f"  {token:<20} -> {', '.join(sugg) if sugg else '(no suggestion)'}")


if __name__ == "__main__":
    main()
