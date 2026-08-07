#!/usr/bin/env python3
"""Generate installers/ChiShona-Word.dic (MS Word custom dictionary, UTF-16LE)
from dictionaries/sn_ZW.{aff,dic}. Bounded expansion to stay Word-friendly."""
from pathlib import Path
import re

root = Path(__file__).resolve().parent.parent
aff = (root / 'dictionaries/sn_ZW.aff').read_text()
dic_lines = (root / 'dictionaries/sn_ZW.dic').read_text().splitlines()[1:]

V = re.findall(r'PFX V 0 (\S+) \.', aff)
C = re.findall(r'PFX C 0 (\S+) \.', aff)
EXT = re.findall(r'SFX E a (\S+) a', aff)
OC = ['ndi', 'ku', 'mu', 'va', 'ti', 'zvi']

words = set()
for line in dic_lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    w, _, flags = line.partition('/')
    words.add(w)
    if 'V' in flags:
        stems = [w] + ([w[:-1] + e for e in EXT] if 'E' in flags and w.endswith('a') else [])
        neg = [w[:-1] + 'i'] if 'P' in flags and w.endswith('a') else []
        for s in stems:
            words.update({'ku' + s, 'kusa' + s})
            words.update(p + s for p in V)
        for s in neg:
            words.update(p + s for p in V)
        for oc in OC:
            words.update(p + oc + w for p in V)
            words.add('ku' + oc + w)
    elif 'C' in flags:
        words.update(p + w for p in C)

out = root / 'installers/ChiShona-Word.dic'
out.write_bytes(b'\xff\xfe' + ("\n".join(sorted(words)) + "\n").encode('utf-16-le'))
print(f"{len(words):,} forms -> {out}")
