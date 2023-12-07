from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import streamlit as st
import soundfile as sf
import numpy as np
import tempfile
import librosa
import torch
import time
import os
import re
import io

load_dotenv()

HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

def check_gpu_availability():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available on this device.")
        print(f"PyTorch version: {torch.__version__}")

        # Displaying number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Displaying information about each GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Setting default CUDA device (optional, for multi-GPU environments)
        torch.cuda.set_device(0)  # Sets the default GPU as GPU 0
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
    else:
        print("CUDA (GPU support) is not available on this device.")

check_gpu_availability()

st.set_page_config(
    page_title="Whisper + Diarization",
    page_icon="🤫",
    layout="centered",
    initial_sidebar_state="expanded",
)

max_new_tokens = st.sidebar.slider("Max New Tokens", 0, 4096, 2048)
chunk_length_s = st.sidebar.slider("Chunk Length (s)", 0, 60, 15)
batch_size = st.sidebar.slider("Batch Size", 0, 128, 64)

# Decorator for caching the Whisper model
@st.cache_resource
def load_whisper_model(model_id, torch_dtype):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model_id = "openai/whisper-large-v3"

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True, use_flash_attention_2=True,
    ).to(device)
    print(f"Model device after moving: {whisper_model.device}")
    whisper_model.to_bettertransformer()
    whisper_processor = AutoProcessor.from_pretrained(model_id)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        max_new_tokens=max_new_tokens,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper_pipe

# Decorator for caching the diarization pipeline
@st.cache_resource
def diarization_pipeline(pipeline_id):
    auth_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            pipeline_id, use_auth_token=auth_token
        ).to(device)
        print(f"Diarization Model device after moving: {diarization_pipeline.device}")
        return diarization_pipeline
    except Exception as e:
        print(f"Failed to load diarization pipeline: {e}")
        return None


def resample_audio(audio, sr, target_sr=16000):
    """
    Resample the given audio to the target sampling rate.

    Parameters:
    audio (numpy.ndarray): The input audio data.
    sr (int): The current sampling rate of the audio.
    target_sr (int): The desired target sampling rate.

    Returns:
    tuple: The resampled audio and the new sampling rate.
    """
    try:
        # Check if the audio is mono or stereo
        if len(audio.shape) > 1:
            # If stereo, average the channels to convert to mono
            audio = np.mean(audio, axis=1)

        # Normalize the audio to -1 to 1 range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Resample the audio if the sampling rates differ
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio, target_sr

    except Exception as e:
        # Handle exceptions and errors
        print(f"Error during resampling: {e}")
        return None, None
    

def transcribe_and_diarize(audio_file_path, language=None, translate=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Load and resample audio for Whisper
        audio, sr = librosa.load(audio_file_path, sr=None)
        resampled_audio, resampled_sr = resample_audio(audio, sr)
        if resampled_audio is None or resampled_sr is None:
            raise ValueError("Failed to resample audio.")

        waveform_tensor_float32 = torch.tensor(resampled_audio, dtype=torch.float32).to(device)
        
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        if translate:
            generate_kwargs["task"] = "translate"
        
        # Load the diarization pipeline from cache and run on audio file
        diarization = diarization_pipeline("pyannote/speaker-diarization-3.1")
        if diarization is None:
            raise ValueError("Failed to load diarization pipeline.")
        
        diarization.to(device)
        diarized_result = diarization(audio_file_path)
        if not diarized_result:
            raise ValueError("Failed to perform diarization.")

        # Load the Whisper pipeline from cache
        whisper_pipe = load_whisper_model("openai/whisper-large-v3", torch.float16)
        if whisper_pipe is None:
            raise ValueError("Failed to load Whisper pipeline.")

        # Constants for chunking
        CHUNK_DURATION, OVERLAP_DURATION = 30, 5  # in seconds
        combined_transcriptions = []

        # Process each turn in the diarization result
        for turn, _, speaker in diarized_result.itertracks(yield_label=True):
            speaker_transcription = ""
            speaker_timestamps = []

            for chunk_start in range(int(turn.start * resampled_sr), int(turn.end * resampled_sr), int(CHUNK_DURATION * resampled_sr)):
                chunk_end = min(chunk_start + int(CHUNK_DURATION * resampled_sr), int(turn.end * resampled_sr))

                speaker_audio_segment = waveform_tensor_float32[chunk_start:chunk_end].to(torch.float16).to(device)
                speaker_audio_segment_np = speaker_audio_segment.cpu().numpy()

                # Process the audio segment with Whisper pipeline
                result = whisper_pipe(
                    speaker_audio_segment_np,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs
                )

                if not result or 'text' not in result:
                    continue

                # Aggregate transcribed text and timestamps for each chunk
                speaker_transcription += result['text'] + " "
                speaker_timestamps.extend(result['chunks'])

            # Append the aggregated transcription and timestamps for the entire turn
            combined_transcriptions.append((speaker, speaker_transcription.strip(), speaker_timestamps))

        return combined_transcriptions

    except Exception as e:
        st.error(f"Error in processing: {e}")
        return []


def rename_speakers(transcriptions, new_names):
    renamed_transcriptions = []
    for speaker, text, timestamps in transcriptions:
        # Using regex to find and replace speaker names
        for old_name, new_name in new_names.items():
            if new_name:  # Check if a new name is provided
                speaker = re.sub(f"^{old_name}$", new_name, speaker)
        renamed_transcriptions.append((speaker, text, timestamps))
    return renamed_transcriptions

def create_subtitle_content(transcriptions):
    subtitle_content = io.StringIO()
    counter = 1
    for speaker, _, timestamps in transcriptions:
        for timestamp in timestamps:
            start_time = format_timestamp(timestamp['timestamp'][0])
            end_time = format_timestamp(timestamp['timestamp'][1])
            subtitle_text = f"{speaker}: {timestamp['text']}"
            subtitle_content.write(f"{counter}\n")
            subtitle_content.write(f"{start_time} --> {end_time}\n")
            subtitle_content.write(f"{subtitle_text}\n\n")
            counter += 1
    return subtitle_content.getvalue()

def format_timestamp(seconds):
    # Converts a timestamp in seconds to the SRT format (HH:MM:SS,MS)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"



def main():
    st.title("Whisper + Diarization")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        audio_file_path = f"sounds/{uploaded_file.name}"
        with open(audio_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        start_time = time.time()

        transcriptions = transcribe_and_diarize(audio_file_path, language="english", translate=False)

        new_speaker_names = {}
        for i in range(5):
            speaker_label = f"SPEAKER_{i:02d}"
            new_name = st.sidebar.text_input(f"Rename {speaker_label}", "", key=f"speaker_{i}")
            new_speaker_names[speaker_label] = new_name
        
        transcriptions = rename_speakers(transcriptions, new_speaker_names)
        end_time = time.time()
        duration = end_time - start_time

        st.info(f"Transcription and diarization took {duration:.2f} seconds")

        for speaker, text, timestamps in transcriptions:
            st.write(f"{speaker}: {text}.")

        user_filename = st.sidebar.text_input("Enter a filename", "transcriptions")

        if st.sidebar.button("Export to Subtitles"):
            # Ensure the filename ends with .srt
            if not user_filename.endswith(".srt"):
                user_filename += ".srt"

            subtitle_content = create_subtitle_content(transcriptions)
            st.sidebar.download_button(
                label="Download Subtitles",
                data=subtitle_content,
                file_name=user_filename,
                mime='text/plain'
            )

if __name__ == "__main__":
    main()



