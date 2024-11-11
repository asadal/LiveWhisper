from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch
import sys
import tempfile as tf
import os
import streamlit as st

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small", device=device
)

transcriber.model.generation_config.language = None
transcriber.model.generation_config.task = None

def create_temp_dir():
    # Create a temporary directory
    set_temp_dir = tf.TemporaryDirectory()
    temp_dir = set_temp_dir.name + "/"
    # 디렉터리 접근 권한 설정
    os.chmod(temp_dir, 0o700)
    return temp_dir

def transcribe(chunk_length_s=10.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    st.write("Start speaking...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        # 'text' 필드가 있는지 확인 후 출력
        try:
            print(item["text"], end="\r")
            if not item.get("partial", [True])[0]:
                break
        except KeyError:
            print("No 'text' field in item. Skipping...")
            continue

    # 오류가 발생하지 않았다면 정상적으로 item["text"] 반환
    return item.get("text", "No text available")

st.set_page_config(
    page_title="Realtime Transcription",
    page_icon="https://static.thenounproject.com/png/48407-200.png"
)
# Featured image
st.image(
    "https://static.thenounproject.com/png/48407-200.png",
    width=150
)
def main():# Main title and description
    st.title("Realtime Transcription")
    st.markdown("Just Press 'Start' and start speaking.")

    start_button = st.button("Start")
    stop_button = st.button("Stop")

    transcript_content = ""
    temp_dir = ""
    
    if start_button:
        # temp_dir = create_temp_dir()
        # transcript_path = temp_dir + "transcript.txt"
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)
        # with open(transcript_path, "a+") as f:
        while True:
            text = transcribe()
            st.write(text)
            # f.write(text + "\n")
            transcript_content += text + "\n"
            if stop_button:
                break
                
    if stop_button:
        st.write("Stopped.")
        st.text_area(transcript_content)
        st.spinner("Downloading transcript...")
        st.download_button(
            label="Donwload Transcript",
            data=transcript_content,
            file_name="transcript.txt",
            mime="text/plain",
        )
        # else:
        #     st.error("No transcript found.")

if __name__ == "__main__":
    main()
