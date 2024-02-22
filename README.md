# Whisper API

## Features
- [x] OpenAI [audio API](https://platform.openai.com/docs/api-reference/audio) format, including both **transcription** and **translation**
- [x] **Insanely fast, even on Mac**, thanks to **[insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)**'s idea
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
   (Optional:) You **MAY** need to install **flash-attention** following [this](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) to get even better performance
3. - API:
     - Easy way:
     ```bash
     chmod +x run_api.sh
     ./run_api.sh
     ```
     - Or:
     ```bash
     python app.py -h
     
     usage: app.py [-h] [--device-id DEVICE_ID] [--model-name MODEL_NAME] [--batch-size BATCH_SIZE] [--chunk-length-s CHUNK_LENGTH_S] [--flash FLASH] --port PORT
              [--concurrent CONCURRENT] [--wait-timeout WAIT_TIMEOUT]

      Automatic Speech Recognition
      
      options:
        -h, --help            show this help message and exit
        --device-id DEVICE_ID
                              Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")
        --model-name MODEL_NAME
                              Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)
        --batch-size BATCH_SIZE
                              Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)
        --chunk-length-s CHUNK_LENGTH_S
                              The length of each ASR chunk. (default: 30)
        --flash FLASH         Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)
        --port PORT           HTTP listening port
        --concurrent CONCURRENT
                              Max concurrency
        --wait-timeout WAIT_TIMEOUT
                              Request max waiting time (in second)
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

     usage: gui.py [-h] [--device-id DEVICE_ID] [--model-name MODEL_NAME] [--batch-size BATCH_SIZE] [--chunk-length-s CHUNK_LENGTH_S] [--flash FLASH] [--port PORT]

      Automatic Speech Recognition
      
      options:
        -h, --help            show this help message and exit
        --device-id DEVICE_ID
                              Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")
        --model-name MODEL_NAME
                              Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)
        --batch-size BATCH_SIZE
                              Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)
        --chunk-length-s CHUNK_LENGTH_S
                              The length of each ASR chunk. (default: 30)
        --flash FLASH         Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)
        --port PORT           Gradio listening port
     ```
     - Default port -> 7860 (web page)
