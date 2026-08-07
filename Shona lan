# Contributing to the Shona Language Project

## For Shona speakers (no coding needed)
The dictionary was seeded by machine and NEEDS native review. Open a
[word report issue](../../issues/new?template=word-report.yml) for any word
that is wrong, missing, or badly suggested. Dialect notes welcome
(Zezuru, Karanga, Manyika, Korekore, Ndau).

## For developers

Requirements: Python 3.10+, `hunspell` (for testing), no other dependencies
for the core tools.

```bash
# test the dictionary
echo "ndinokutenda kubatsirana" | hunspell -d dictionaries/sn_ZW -l   # blank = pass

# grow the lexicon from a corpus
python3 tools/build_wordlist.py your_shona_text.txt --min-count 3 \
    --dic dictionaries/sn_ZW.dic --out grown.dic

# run the autocorrect demo
python3 tools/autocorrect.py --dic dictionaries/sn_ZW.dic \
    --aff dictionaries/sn_ZW.aff "zvakanka mhorooi"
```

### Adding words
- Verb stems go in `dictionaries/sn_ZW.dic` with flags `/VNPOE`
  (V = takes subject/tense prefixes, N = takes ku-, P = negative -i,
  O = takes object concords, E = takes verbal extensions)
- Nouns/other words: plain, or `/C` if they take na-/sa-/mu-/pa-/ku- particles
- Update the count on line 1, then rebuild the installers (see below)

### Rebuilding installers after dictionary changes
```bash
cd installers && ./rebuild.sh
```

### Pull requests
Keep PRs focused (lexicon PRs separate from tooling PRs). For lexicon
changes, cite a source or provide example sentences for new words.

## Speech recognition (Phase 3)
`tools/finetune_whisper_shona.py` trains on Google's WAXAL Shona data —
free Colab GPU is enough for whisper-small. Share resulting WER numbers
in an issue; attribute WAXAL (CC-BY-4.0) in any model release.
