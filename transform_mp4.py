import subprocess
import os
from pathlib import Path

def mp4_to_wav(input_path, output_path=None, sr=16000):
    """
    Convert .mp4 to .wav using FFmpeg with ASR-optimized parameters.
    - input_path: path to .mp4 file
    - output_path: destination .wav (auto-generated if None)
    - sr: desired sampling rate (default 16 kHz for ASR)

    Output WAV:
        - PCM S16LE
        - Mono
        - 16 kHz (or custom)
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",                     # overwrite output
        "-i", str(input_path),    # input file
        "-ac", "1",               # mono
        "-ar", str(sr),           # sample rate
        "-f", "wav",              # wav format
        "-acodec", "pcm_s16le",   # 16-bit PCM
        str(output_path),
    ]

    # Execute FFmpeg
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return str(output_path)

if __name__ == "__main__":
    wav_path = mp4_to_wav(r"D:\Coding Programs\RSMM\f2_test.mp4")
    print("Converted to:", wav_path)