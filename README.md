# Whisper API

## Features
- [x] OpenAI [audio API](https://platform.openai.com/docs/api-reference/audio) format, including both **transcription** and **translation**
- [x] **Insanely fast**, thanks to **[insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)**'s idea
- [x] Support **prompt**
- [x] Support response in **json/text/srt/vtt**
- [x] **json** response provides **chunks** as well
- [x] Support input from **URL**
- [x] Don't like API? **Try the easy-to-use GUI!**

## Usage
1. Clone this repository
2. ```bash
   conda create -n whisperapi python=3.10 -y
   conda activte whisperapi
   pip install -r requirements.txt
   ```
3. - API:
     - Easy way:
     ```bash
     chmod +x run_api.sh
     ./run_api.sh
     ```
     - Or:
     ```bash
     python app.py -h
     ```
     - Default port -> 9000 (HTTP POST API):
     ```bash
     curl http://127.0.0.1:9000/v1/audio/transcriptions \
      -H "Content-Type: multipart/form-data" \
      -F file="@/path/to/file/audio.mp3" \
      -F model="whisper-1"
     ```
   - GUI:
     - Easy way:
     ```bash
     chmod +x run_gui.sh
     ./run_gui.sh
     ```
     - Or:
     ```bash
     python gui.py -h
     ```
     - Default port -> 7860 (web page)
