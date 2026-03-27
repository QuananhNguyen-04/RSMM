import subprocess
import os
import sys
from pathlib import Path


def mp3_to_wav(input_path, output_path=None, sr=16_000):
    try:
        
        input_path = Path(input_path)
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
    
        if output_path is None:
            output_path = input_path.with_suffix(".wav")
        else:
            output_path = Path(output_path)

        # Run ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", str(sr),
            "-f", "wav",
            str(output_path),
        ]

        subprocess.run(cmd)
        print(f"Conversion successful: {output_path}")

    except subprocess.CalledProcessError:
        print("ffmpeg failed to convert the file.")
    except Exception as e:
        print(f"Error: {e}")
    if not output_path.exists():
        raise RuntimeError("Conversion failed: WAV not created.")
    return str(output_path)


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python mp3_to_wav_ffmpeg.py input.mp3 output.wav")
    #     sys.exit(1)

    mp3_to_wav(
        "D:\Coding Programs\RSMM\LLM hype led nowhere back to classic methods.mp3"
    )
