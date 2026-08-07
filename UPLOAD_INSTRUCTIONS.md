# PUTTING THIS ON GITHUB — two ways (pick one)

Your repository: github.com/<your-username>/shona-language
(If you named it "shona language" with a space, GitHub will have made it
"shona-language" — check the URL in your browser.)

---------------------------------------------------------------
OPTION A — Command line (best; 3 commands, needs git installed)
---------------------------------------------------------------
1. Unzip this folder on your computer, open a terminal INSIDE the
   shona-language folder, then run:

   git init
   git add .
   git commit -m "Shona Digital Toolkit v0.2: morphology-aware spellchecker, tools, WAXAL integration"
   git branch -M main
   git remote add origin https://github.com/<your-username>/shona-language.git
   git push -u origin main

2. When prompted to log in, use your GitHub username and a Personal Access
   Token as the password (GitHub → Settings → Developer settings →
   Personal access tokens → Generate). Keep the token private — never
   share it with anyone, including AI chats.

---------------------------------------------------------------
OPTION B — Web upload (no git needed, works from any browser)
---------------------------------------------------------------
1. Open your repository page on github.com
2. Click "Add file" → "Upload files"
3. Drag in ALL the contents of this folder (not the folder itself):
   README.md, LICENSE, CONTRIBUTING.md, .gitignore,
   and the dictionaries/, tools/, installers/, docs/ folders
4. Commit. Then repeat once more for the .github folder if your browser
   skipped it (hidden folders sometimes don't drag) — you can also create
   the files manually: "Add file" → "Create new file" → type
   `.github/ISSUE_TEMPLATE/word-report.yml` as the filename and paste
   the contents.

---------------------------------------------------------------
AFTER UPLOADING (5 minutes, big payoff)
---------------------------------------------------------------
1. Repo → Settings → General → Features: make sure "Issues" is ON
   (the word-report template turns Shona speakers into contributors)
2. Repo main page → About (right side, gear icon):
   - Description: "Free Shona spellchecker, autocorrect & speech tools —
     bringing ChiShona into the digital era"
   - Topics: shona, zimbabwe, nlp, hunspell, spellchecker,
     african-languages, low-resource-languages, speech-recognition
   (Topics are how Masakhane researchers and funders will FIND you)
3. Create your first Release: Releases → "Create a new release" →
   tag v0.2.0 → attach installers/shona-spellcheck-0.2.oxt and
   installers/ChiShona-Word.dic → publish.
   (Releases give people a one-click download without browsing code.)
4. Share the repo link in the Masakhane community (masakhane.io → join
   Slack) and in your university emails — a live repo with CI passing
   is worth ten proposals.

The GitHub Actions test workflow (.github/workflows/test.yml) runs
automatically on every push: it checks that valid Shona passes and
gibberish fails, so contributions can't silently break the dictionary.
