import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Replace with your Hugging Face access token
huggingface_token = "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"

# Instantiate the pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=huggingface_token)

# Send pipeline to GPU
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
else:
    print("CUDA is not available. Running on CPU.")

# Load and process the audio file
# This can be replaced with the "processing from memory" approach if needed
audio_file = "$UICIDEBOY$ - NOT EVEN GHOSTS ARE THIS EMPTY.mp3"
with ProgressHook() as hook:
    diarization = pipeline(audio_file, hook=hook)

# Optionally, control the number of speakers
# diarization = pipeline(audio_file, num_speakers=2)
# diarization = pipeline(audio_file, min_speakers=2, max_speakers=5)

# Dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm_file:
    diarization.write_rttm(rttm_file)

print("Diarization completed and saved to 'audio.rttm'.")


def transcribe_and_diarize(audio_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Load and resample audio for Whisper
        audio, sr = librosa.load(audio_file_path, sr=None)
        resampled_audio, resampled_sr = resample_audio(audio, sr)
        if resampled_audio is None or resampled_sr is None:
            raise ValueError("Failed to resample audio.")

        waveform_tensor_float32 = torch.tensor(resampled_audio, dtype=torch.float32).to(device)

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
            for chunk_start in range(int(turn.start * resampled_sr), int(turn.end * resampled_sr), int(CHUNK_DURATION * resampled_sr)):
                chunk_end = min(chunk_start + int(CHUNK_DURATION * resampled_sr), int(turn.end * resampled_sr))
                speaker_audio_segment = waveform_tensor_float32[chunk_start:chunk_end].to(torch.float16).to(device)
                speaker_audio_segment_np = speaker_audio_segment.cpu().numpy()

                # Process the audio segment with Whisper pipeline
                result = whisper_pipe({'raw': speaker_audio_segment_np, 'sampling_rate': resampled_sr})
                if not result or 'text' not in result:
                    raise ValueError("Failed to transcribe audio segment.")

                combined_transcriptions.append((speaker, result['text']))

                if chunk_end == int(turn.end * resampled_sr):
                    break

        return combined_transcriptions

    except Exception as e:
        st.error(f"Error in processing: {e}")
        return []