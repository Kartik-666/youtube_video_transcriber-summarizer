import whisper
import numpy as np
from pydub import AudioSegment

def transcribe_audio(audio_input):
    """Transcribes audio using Whisper. Accepts file path or pydub.AudioSegment."""
    model = whisper.load_model("base.en")

    if isinstance(audio_input, AudioSegment):
        # Convert AudioSegment to NumPy array
        samples = np.array(audio_input.get_array_of_samples()).astype(np.float32) / (1 << 15)
        result = model.transcribe(samples)
    else:
        result = model.transcribe(audio_input)

    return result["text"]
