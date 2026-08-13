# Shona Language — Digital Infrastructure for ChiShona 🇿🇼

**Bringing ChiShona into the digital era.** Spellcheck, autocorrect, and speech
recognition for 10+ million Shona speakers — free, open source, MIT-licensed.

> Type Shona without red squiggles. Autocorrect that suggests *Shona*, not English.
> No more "language detected: Indonesian". **Live Shona speech recognition is now real.**

[![Shona ASR Model](https://img.shields.io/badge/🤗%20Hugging%20Face-whisper--small--shona-blue)](https://huggingface.co/Starsm91/whisper-small-shona)
[![WER: 36.42%](https://img.shields.io/badge/WER-36.42%25-green)](https://huggingface.co/Starsm91/whisper-small-shona)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Install the spellchecker today

**LibreOffice / OpenOffice (free):**
Download [`installers/shona-spellcheck-0.2.oxt`](installers/shona-spellcheck-0.2.oxt)
and double-click it. Then set your text language: *Tools → Language → For All Text → More → Shona*.

**Microsoft Word (Windows):**
1. Download [`installers/ChiShona-Word.dic`](installers/ChiShona-Word.dic) (214,677 Shona word forms)
2. Copy it to `C:\Users\<you>\AppData\Roaming\Microsoft\UProof\`
3. In Word: *File → Options → Proofing → Custom Dictionaries → Add*, select it,
   set *Dictionary language: All languages*
4. To stop autocorrect mangling Shona: *File → Options → Proofing → AutoCorrect
   Options* → untick *Replace text as you type* while writing Shona

**Firefox / Chrome / Linux / macOS:** copy `dictionaries/sn_ZW.aff` and
`sn_ZW.dic` into your system Hunspell folder (`/usr/share/hunspell/` on Linux,
`~/Library/Spelling/` on macOS).

## 🧠 How it works

Shona is agglutinative — one verb stem produces hundreds of valid words
(*ndinotenda, vachatenda, handitendi, ndinokutenda, kutendwa...*). A flat
wordlist can never keep up, so this project encodes the **morphology itself**
as Hunspell rules:

| Layer | Example | Status |
|---|---|---|
| Subject concords + tense (78 complexes) | **ndicha**enda, **vaka**uya | ✅ v0.1 |
| Object concords (two-level prefixes) | ndino**ku**da, vaka**ndi**udza | ✅ v0.2 |
| Verbal extensions (-ir/-is/-w/-an/-ik/-isw) | kubatsir**an**a, akashand**isw**a | ✅ v0.2 |
| Negative paradigm | **handi**tend**i** | ✅ v0.2 |
| 603 curated seed stems → | **214,677 recognised forms** | growing |


## 🎙 Shona Speech Recognition (NEW)

The first open Shona speech recognition model is now live:

**[huggingface.co/Starsm91/whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona)**

- Trained on Google's WAXAL Shona ASR dataset (CC-BY-4.0)
- Word Error Rate: **36.42%** — roughly 2 in 3 words correct, first version
- Based on OpenAI Whisper-small, fine-tuned for chiShona
- Use it in Python: `pipeline("automatic-speech-recognition", model="Starsm91/whisper-small-shona")`
- Foundation for live TV and video captioning in Shona

## 🛠 Tools

| Script | Purpose |
|---|---|
| `tools/build_wordlist.py` | Feed it any Shona text (news, Bible, Wikipedia) → grows the dictionary with real vocabulary + frequency data |
| `tools/autocorrect.py` | Prototype autocorrect engine with Shona-tuned edit costs (l/r, bh/b, vh/v) |
| `tools/waxal_to_corpus.py` | Harvest Shona transcripts from [Google's WAXAL dataset](https://huggingface.co/datasets/google/WaxalNLP) as a text corpus |
| `tools/finetune_whisper_shona.py` | Fine-tune Whisper for Shona speech recognition on WAXAL — runs on a **free Google Colab GPU** |

## 🗺 Roadmap

- [x] **v0.1** — Seed lexicon, SC/TAM verb morphology, LibreOffice + Word packaging
- [x] **v0.2** — Object concords, verbal extensions, negatives; WAXAL integration scripts
- [x] **v0.5** — **[DONE]** Shona Whisper ASR model published: [huggingface.co/Starsm91/whisper-small-shona](https://huggingface.co/Starsm91/whisper-small-shona) — WER: **36.42%** (trained on Google WAXAL, 4000 steps, ~13,000 examples)
- [ ] **v0.3** — Native-speaker validated lexicon (help us! see below), corpus-grown to 50k entries; Firefox/Chrome/Android dictionary submissions
- [ ] **v0.4** — Android keyboard (HeliBoard fork) with Shona autocorrect + prediction
- [ ] **v0.6** — Fine-tune ASR further on Zimbabwean broadcast audio; deploy captioning pipeline
- [ ] **v1.0** — University-endorsed release; LibreOffice official dictionary repo inclusion

## 🤝 Contribute (especially if you speak Shona!)

**You don't need to code.** The most valuable contribution is native-speaker
review:

- 🐛 **Found a wrong word or a missing word?**
  [Open a word report](../../issues/new?template=word-report.yml) — takes 30 seconds.
- 📚 Have Shona text (articles, stories, transcripts you have rights to share)?
  Open an issue — every text grows the dictionary.
- 💻 Developers: see [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup.
- 🎓 Universities & researchers: see
  [`docs/Shona-Digital-Language-Infrastructure-Proposal.docx`](docs/Shona-Digital-Language-Infrastructure-Proposal.docx)
  for the partnership proposal.

## 🌍 Related work we build with

[Masakhane](https://www.masakhane.io/) ·
[Shona spaCy](https://pypi.org/project/shona-spacy/) ·
[WAXAL](https://huggingface.co/datasets/google/WaxalNLP) ·
[Mozilla Common Voice](https://commonvoice.mozilla.org/) ·
ALRI (University of Zimbabwe) ALLEX corpus

## ⚖️ License

MIT — free for everyone, forever. WAXAL data is © Google under CC-BY-4.0;
attribute it in any derived corpus or model release.

---
*Mutauro wedu, panyika yose.* — Our language, everywhere in the world.
