{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Whisper-Shona v2 \u2014 Project Nyaradzai\n",
    "**Disconnect-proof version.** Checkpoints save to your Google Drive, every step verifies itself.\n",
    "\n",
    "**Golden rule: after ANY disconnect or restart, always run Cell 1 again first.** It is safe to re-run everything from the top at any time \u2014 completed training resumes where it left off.\n",
    "\n",
    "Setup once: Runtime \u2192 Change runtime type \u2192 **GPU** \u2192 Save."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 1 \u2014 Setup (ALWAYS run this first, and again after any disconnect)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# 1a. Connect your Google Drive (checkpoints survive disconnects there)\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "import os\n",
    "WORKDIR = '/content/drive/MyDrive/whisper-shona'\n",
    "os.makedirs(WORKDIR, exist_ok=True)\n",
    "print('Model checkpoints will be stored in:', WORKDIR)\n",
    "\n",
    "# 1b. Install libraries\n",
    "!pip install -q transformers datasets evaluate jiwer accelerate soundfile librosa\n",
    "\n",
    "# 1c. Download the training script \u2014 with LOUD verification\n",
    "import urllib.request\n",
    "url = 'https://raw.githubusercontent.com/stanleymateta-tech/Project-Nyaradzai/main/tools/finetune_whisper_shona.py'\n",
    "urllib.request.urlretrieve(url, '/content/finetune_whisper_shona.py')\n",
    "size = os.path.getsize('/content/finetune_whisper_shona.py')\n",
    "assert size > 3000, 'Download failed \u2014 check the URL / your repo'\n",
    "print(f'Training script downloaded OK ({size:,} bytes)')\n",
    "\n",
    "# 1d. Confirm GPU\n",
    "import torch\n",
    "if torch.cuda.is_available():\n",
    "    print('GPU ready:', torch.cuda.get_device_name(0))\n",
    "else:\n",
    "    print('*** NO GPU! Runtime -> Change runtime type -> GPU, then re-run this cell ***')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 2 \u2014 Log in to Hugging Face\n",
    "(huggingface.co \u2192 Settings \u2192 Access Tokens \u2192 New token, type **Write**. Needed once per session.)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from huggingface_hub import notebook_login\n",
    "notebook_login()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 3 \u2014 Train\n",
    "Takes ~2\u20134 hours. Safe to interrupt: because checkpoints are on Drive, re-running Cell 1 then this cell **resumes automatically** \u2014 you never lose progress."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python /content/finetune_whisper_shona.py --model openai/whisper-small --output /content/drive/MyDrive/whisper-shona --max-steps 4000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 4 \u2014 Publish to Hugging Face\n",
    "(Checks everything before uploading and tells you exactly what's wrong if something is missing.)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import os\n",
    "from huggingface_hub import HfApi\n",
    "\n",
    "MODEL_DIR = '/content/drive/MyDrive/whisper-shona'\n",
    "\n",
    "# pre-flight checks with friendly messages\n",
    "if not os.path.isdir(MODEL_DIR):\n",
    "    raise SystemExit('Model folder not found. Run Cell 1, then Cell 3, first.')\n",
    "if not any(f.endswith('.safetensors') or f == 'pytorch_model.bin' for f in os.listdir(MODEL_DIR)):\n",
    "    raise SystemExit('No trained model weights in the folder yet \u2014 Cell 3 has not finished. '\n",
    "                     'If it was interrupted, re-run Cell 1 then Cell 3 (it resumes automatically).')\n",
    "\n",
    "api = HfApi()\n",
    "repo_id = api.whoami()['name'] + '/whisper-small-shona'\n",
    "api.create_repo(repo_id, exist_ok=True)\n",
    "api.upload_folder(folder_path=MODEL_DIR, repo_id=repo_id,\n",
    "                  ignore_patterns=['checkpoint-*'])   # upload final model only, not checkpoints\n",
    "print('PUBLISHED: https://huggingface.co/' + repo_id)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 \u2014 Test it on real audio\n",
    "Upload a Shona .mp3/.wav using the folder icon on the left, put its name below, and run."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import torch\n",
    "from transformers import pipeline\n",
    "asr = pipeline('automatic-speech-recognition',\n",
    "               model='/content/drive/MyDrive/whisper-shona',\n",
    "               device=0 if torch.cuda.is_available() else -1)\n",
    "print(asr('your_audio.mp3', return_timestamps=True)['text'])"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
