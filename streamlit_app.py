from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch
import sys
import tempfile as tf
import os
import streamlit as st

# 디바이스 설정
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 자동 음성 인식 파이프라인 초기화
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small", device=device
)
transcriber.model.generation_config.language = None
transcriber.model.generation_config.task = None

def create_temp_dir():
    set_temp_dir = tf.TemporaryDirectory()
    temp_dir = set_temp_dir.name + "/"
    os.chmod(temp_dir, 0o700)
    return temp_dir

# 음성을 실시간으로 전사하는 함수
def transcribe(chunk_length_s=10.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    # 마이크 스트림 시작
    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    # 'Start speaking...' 메시지를 한 번만 출력
    first_run = True

    # 음성이 입력될 때까지 대기 상태 유지
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        # 최초 실행 시 메시지 출력
        if first_run:
            st.write("Start speaking...")
            first_run = False  # 이후로는 이 메시지가 반복되지 않음

        # item에 text 필드가 있는지 확인하여 음성 입력이 없으면 루프 계속
        if "text" in item and item["text"].strip():
            sys.stdout.write("\033[K")  # 콘솔에서 이전 텍스트 삭제
            print(item["text"], end="\r")
            return item["text"]  # 텍스트 반환

    # 음성이 없는 경우 None 반환하지 않고 대기 유지
    return None

# Streamlit 인터페이스 설정
st.set_page_config(
    page_title="Realtime Transcription",
    page_icon="https://static.thenounproject.com/png/48407-200.png"
)

st.image(
    "https://static.thenounproject.com/png/48407-200.png",
    width=150
)

def main():
    # 타이틀 및 설명 표시
    st.title("Realtime Transcription")
    st.markdown("Just Press 'Start' and start speaking.")

    start_button = st.button("Start")
    stop_button = st.button("Stop")

    transcript_content = ""
    
    # Start 버튼이 눌렸을 때 실시간 전사 시작
    if start_button:
        while True:
            text = transcribe()
            if text:  # 텍스트가 반환된 경우에만 출력
                st.write(text)
                transcript_content += text + "\n"
            if stop_button:
                break

    # Stop 버튼이 눌렸을 때 텍스트 저장 및 다운로드
    if stop_button:
        st.write("Stopped.")
        st.text_area("Transcript", transcript_content)
        st.download_button(
            label="Download Transcript",
            data=transcript_content,
            file_name="transcript.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()
