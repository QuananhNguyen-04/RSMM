import json
from typing import Tuple
import torch
import torchaudio
import os
import numpy as np
import re
import numbers

def dump_dataset_paths(ds, out_dir, n=10):
    os.makedirs(out_dir, exist_ok=True)

    for i, row in enumerate(ds.select(range(n))):
        sample_dir = os.path.join(out_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # save only the path reference
        audio_path = row["audio"]["path"]

        # save metadata (all other features + audio path)
        meta = {k: v for k, v in row.items() if k != "audio"}
        meta["audio_path"] = audio_path

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved {n} sample metadata entries to {out_dir}")

def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    # Case 1: flat vector [num_samples]
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    # Case 2: stereo or multi-channel [channels, num_samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # collapse to mono

    return waveform


def load_audio(file_path, target_sr=16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file (wav, mp3, mp4, etc.) and resample if needed.
    Returns: waveform tensor, sample_rate
    """
    waveform, sr = torchaudio.load(file_path)

    # resample if needed
    if target_sr and sr != target_sr:
        print(f"resample the {file_path} from {sr} to {target_sr}")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    waveform = ensure_mono(waveform)
    return waveform, sr

def is_titanet(model) -> bool:
    """
    Heuristic check if a model is Titanet.
    Works for NeMo Titanet (EncDecSpeakerLabelModel).
    """
    name = model.__class__.__name__.lower()
    mod = model.__class__.__module__.lower()
    return "titanet" in name or "nemo" in mod

def pad_or_trim(arr, target_len, fill=None):
    """Pad or trim a 1D array to target_len."""
    arr = np.asarray(arr)
    cur_len = len(arr)
    if cur_len == target_len:
        return arr
    if cur_len < target_len:
        pad = np.full(target_len - cur_len, fill, dtype=object)
        return np.concatenate([arr, pad])
    return arr[:target_len]

def segments_to_frames(segments, speakers, frame_len=0.01):
    """
    Convert segment-level annotations to frame-level speaker IDs.
    Returns frames (np.ndarray) and spk_to_id (dict).
    """
    assert len(segments) == len(speakers), "Segments and speakers must align"
    unique_speakers = list(dict.fromkeys(speakers))
    spk_to_id = {spk: i for i, spk in enumerate(unique_speakers)}
    max_time = max(end for _, end in segments)
    T = int(np.ceil(max_time / frame_len))
    frames = np.full(T, None)
    for (start, end), spk in zip(segments, speakers):
        sid = spk_to_id[spk]
        start_idx = int(np.floor(start / frame_len))
        end_idx = int(np.ceil(end / frame_len))
        frames[start_idx:end_idx] = sid
    return frames, spk_to_id

def expand_segments_to_frames(segments, speakers, frame_len=0.01):
    """Convert (start, end) segments with speaker labels into frame-level labels."""
    T = int(max(e for _, e in segments) / frame_len) + 1
    frames = np.array([None] * T)
    for (s, e), spk in zip(segments, speakers):
        s_idx = int(s / frame_len)
        e_idx = int(e / frame_len)
        frames[s_idx:e_idx] = spk
    return frames

def group_runs(seq):
    """Collapse into (value, run_length) ignoring None."""
    runs, prev, count = [], None, 0
    for x in seq:
        if x is None: continue
        if x != prev:
            if prev is not None: runs.append((prev, count))
            prev, count = x, 1
        else:
            count += 1
    if prev is not None: runs.append((prev, count))
    return runs

def extract_frame_errors(gt, sys_remap, frame_len=0.01, window=5, merge_gap=0.5):
    """
    Extract mismatched segments (GT vs SYS) with local neighborhood.
    Returns list of error intervals.
    """
    T = len(gt)
    raw = []
    for i in range(T):
        if gt[i] is None: continue
        if sys_remap[i] != gt[i]:
            for j in range(max(0, i - window), min(T, i + window + 1)):
                if gt[j] is not None:
                    raw.append((j * frame_len, gt[j], sys_remap[j]))
    if not raw: return []
    raw.sort(key=lambda x: x[0])
    start_t, g0, s0 = raw[0]
    end_t = start_t
    errors = []
    for t, g, s in raw[1:]:
        if g == g0 and s == s0 and t - end_t <= merge_gap:
            end_t = t
        else:
            errors.append({"start_time": round(start_t, 2), "end_time": round(end_t, 2), "gt": g0, "sys": s0})
            start_t, end_t, g0, s0 = t, t, g, s
    errors.append({"start_time": round(start_t, 2), "end_time": round(end_t, 2), "gt": g0, "sys": s0})
    return errors

def normalize_speaker_label(lbl):
    """Normalize speaker labels like 'Speaker0_tent', 'spk_1', or 2 into an integer index."""
    if isinstance(lbl, numbers.Integral):
        return lbl
    if isinstance(lbl, str):
        # extract the first numeric part if it exists
        m = re.search(r'\d+', lbl)
        if m:
            return int(m.group(0))
    raise ValueError(f"Cannot normalize speaker label: {lbl!r}")

def normalize_labels(labels):
    """Normalize a list of speaker labels into integer indices."""
    return [normalize_speaker_label(lbl) for lbl in labels]

def save_transcriptions_json(transcriptions: list, output_path: str = "transcriptions.json"):
    """
    Persist a list of transcription objects to disk as formatted JSON.

    Parameters
    ----------
    transcriptions : list
        A Python list of dicts, each containing 'start', 'end', 'speaker', and 'text'.
    output_path : str
        Target file path for the JSON output.
    """
    if not isinstance(transcriptions, list):
        raise TypeError("Expected 'transcriptions' to be a list, not a string or other type.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Successfully saved {len(transcriptions)} segments → {output_path}")

    
def concat_json_arrays(raw_output: str):
    """
    Parse a string containing one or multiple JSON arrays (possibly concatenated)
    and return a single merged Python list.

    Parameters
    ----------
    raw_output : str
        The raw string output from diarization postprocessing (may contain multiple arrays).

    Returns
    -------
    list
        Merged list of all transcription segments.
    """
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    # Find all JSON arrays within the text
    arrays = re.findall(r"\[[\s\S]*?\]", cleaned)
    if not arrays:
        raise ValueError("No JSON arrays found in the provided string.")

    merged = []
    for arr in arrays:
        try:
            data = json.loads(arr)
            if isinstance(data, list):
                merged.extend(data)
        except json.JSONDecodeError:
            # Skip malformed chunks silently or log as needed
            continue

    cleaned_merged = []
    for item in merged:
        if not isinstance(item, dict):
            continue

        start_ok = bool(item.get("start"))
        end_ok = bool(item.get("end"))
        speaker_ok = bool(item.get("speaker"))
        text_ok = bool(item.get("text"))

        if start_ok and end_ok and speaker_ok and text_ok:
            cleaned_merged.append(item)

    return cleaned_merged

def time_to_seconds(t):
    """
    Convert time to seconds.
    Supports:
    - float / int
    - "HH:MM:SS"
    - "MM:SS"
    """
    if isinstance(t, (int, float)):
        return float(t)

    if isinstance(t, str):
        parts = t.strip().split(":")
        parts = [float(p) for p in parts]

        if len(parts) == 3:  # HH:MM:SS
            h, m, s = parts
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:  # MM:SS
            m, s = parts
            return m * 60 + s
        else:
            raise ValueError(f"Invalid time format: {t}")

    raise TypeError(f"Unsupported time type: {type(t)}")