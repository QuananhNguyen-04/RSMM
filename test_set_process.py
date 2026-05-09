import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict


# ----------------------------
# Config
# ----------------------------
IGNORE_TAGS = {"disfmarker"}   # non-word annotations
KEEP_PUNCT = True              # keep punctuation tokens


# ----------------------------
# Parse a single words.xml file
# ----------------------------
def parse_words_xml(file_path: str) -> List[Dict]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    filename = os.path.basename(file_path)
    speaker = filename.split(".")[1]

    words = []
    last_word_idx = None  # index in words list

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag in IGNORE_TAGS:
            continue

        if tag == "w":
            text = (elem.text or "").strip()
            if not text:
                continue

            start = elem.attrib.get("starttime")
            end = elem.attrib.get("endtime")
            if start is None or end is None:
                continue

            start = float(start)
            end = float(end)

            is_punc = elem.attrib.get("punc") == "true"

            if is_punc:
                # Attach to previous word of SAME speaker
                if last_word_idx is not None:
                    words[last_word_idx]["word"] += text
                    words[last_word_idx]["end"] = max(words[last_word_idx]["end"], end)
                # else: drop safely
                continue

            # Normal word
            words.append({
                "start": start,
                "end": end,
                "word": text,
                "speaker": speaker
            })

            last_word_idx = len(words) - 1

    return words

# ----------------------------
# Load all speakers for a meeting
# ----------------------------
def load_meeting_words(words_dir: str, meeting_id: str) -> List[Dict]:
    """
    Load all words.xml files for a meeting and merge
    """
    pattern = os.path.join(words_dir, f"{meeting_id}.*.words.xml")
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"No words.xml files found for meeting {meeting_id}")

    all_words = []

    for file_path in files:
        words = parse_words_xml(file_path)
        all_words.extend(words)

    # Sort globally by time
    all_words.sort(key=lambda x: x["start"])

    return all_words


# ----------------------------
# Optional: Build segments
# ----------------------------
def build_segments(words: List[Dict], gap_threshold: float = 0.5) -> List[Dict]:
    """
    Group words into segments based on:
    - same speaker
    - time gap threshold
    """
    if not words:
        return []

    segments = []
    current = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "text": words[0]["word"]
    }

    for w in words[1:]:
        same_speaker = (w["speaker"] == current["speaker"])
        gap = w["start"] - current["end"]

        if same_speaker and gap <= gap_threshold:
            # extend segment
            current["end"] = w["end"]
            current["text"] += " " + w["word"]
        else:
            segments.append(current)
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["word"]
            }

    segments.append(current)
    return segments


# ----------------------------
# Save JSON
# ----------------------------
import json

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ----------------------------
# Main pipeline
# ----------------------------
if __name__ == "__main__":
    from config import Config

    cfg = Config()

    WORDS_DIR = cfg.words_dir
    MEETING_ID = cfg.meeting_id

    words = load_meeting_words(WORDS_DIR, MEETING_ID)

    print(f"Loaded {len(words)} words")

    # Save word-level GT
    save_json(words, f"{MEETING_ID}_words.json")

    # Optional: build segments
    segments = build_segments(words)

    print(f"Built {len(segments)} segments")

    save_json(segments, f"{MEETING_ID}_segments.json")