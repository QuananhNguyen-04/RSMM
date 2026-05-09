from collections import defaultdict
import json
import re
import bisect

# =========================
# 1. TIME NORMALIZATION
# =========================
def time_to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)

    if isinstance(t, str):
        parts = t.strip().split(":")
        parts = [float(p) for p in parts]

        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = parts
            return m * 60 + s

    raise ValueError(f"Invalid time format: {t}")


# =========================
# 2. SPEAKER NORMALIZATION
# =========================
def normalize_speaker(s):
    s = s.strip().lower()

    mapping = {
        "speaker 1": "A",
        "speaker 2": "B",
        "speaker 3": "C",
        "speaker 4": "D",
    }

    return mapping.get(s, s.upper())


# =========================
# 3. TEXT NORMALIZATION
# =========================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


STOPWORDS = {"uh", "um", "mm", "hmm", "uhh", "umm"}

def is_meaningful(word):
    w = word.lower().strip()
    return w.isalpha() and w not in STOPWORDS and (w != "a" and len(w) > 1)


# =========================
# 4. PREPROCESS PRED SEGMENTS
# =========================
def normalize_pred_segments(pred_segments):
    normalized = []

    for seg in pred_segments:
        normalized.append({
            "start": time_to_seconds(seg["start"]),
            "end": time_to_seconds(seg["end"]),
            "speaker": seg["speaker"],
            "text": seg["text"]
        })

    normalized.sort(key=lambda x: x["start"])
    return normalized


# =========================
# 5. BUILD INDEX
# =========================
def build_segment_index(segments):
    starts = [s["start"] for s in segments]
    return starts, segments


def get_active_segment(t, starts, segments):
    idx = bisect.bisect_right(starts, t) - 1

    if 0 <= idx < len(segments):
        seg = segments[idx]
        if seg["start"] <= t <= seg["end"]:
            return seg

    return None


# =========================
# 6. WORD COVERAGE
# =========================
def compute_word_coverage(gt_words, pred_segments):
    seg_starts, pred_segments = build_segment_index(pred_segments)

    total = 0
    covered = 0

    for w in gt_words:
        if not is_meaningful(w["word"]):
            continue

        total += 1

        seg = get_active_segment(w["start"], seg_starts, pred_segments)
        if seg is None:
            print(w)
            continue

        pred_text = normalize_text(seg["text"])
        gt_word = normalize_text(w["word"])

        if gt_word in pred_text:
            covered += 1

    return covered / total if total > 0 else 0.0

def build_speaker_mapping(gt_words, pred_segments):
    seg_starts, pred_segments = build_segment_index(pred_segments)

    mapping_counts = defaultdict(lambda: defaultdict(int))

    for w in gt_words:
        if not is_meaningful(w["word"]):
            continue

        seg = get_active_segment(w["start"], seg_starts, pred_segments)
        if seg is None:
            continue
        # print(seg)
        pred_text = normalize_text(seg["text"])
        gt_word = normalize_text(w["word"])

        # only count if word is actually covered
        if gt_word in pred_text:
            
            mapping_counts[w["speaker"]][seg["speaker"]] += 1
    print(mapping_counts)
    # pick best mapping
    mapping = {}
    for gt_spk, counts in mapping_counts.items():
        best_pred = max(counts.items(), key=lambda x: x[1])[0]
        mapping[gt_spk] = best_pred

    return mapping

# =========================
# 7. SPEAKER ACCURACY
# =========================
def compute_speaker_accuracy(gt_words, pred_segments, mapping):
    seg_starts, pred_segments = build_segment_index(pred_segments)

    total = 0
    correct = 0

    for w in gt_words:
        if not is_meaningful(w["word"]):
            continue

        total += 1

        seg = get_active_segment(w["start"], seg_starts, pred_segments)
        if seg is None:
            continue

        if w["speaker"] in mapping and mapping[w["speaker"]] == seg["speaker"]:
            correct += 1

    return correct / total if total > 0 else 0.0


# =========================
# 8. COMBINED SCORE
# =========================
def combined_score(coverage, speaker_acc):
    return coverage * speaker_acc

if __name__ == "__main__":
    from config import Config

    cfg = Config()

    gt = json.load(open(cfg.gt_path, "r", encoding="utf-8"))
    pred = json.load(open(cfg.pred_path, "r", encoding="utf-8"))
    # Load data
    # gt = json.load(open("./IB4002_words.json", "r", encoding="utf-8"))
    # pred = json.load(open("./transcriptions/final_transcriptions_IB4002.Mix-Headset.json", "r", encoding="utf-8"))
    # normalize prediction timestamps (important)
    pred = normalize_pred_segments(pred)

    # metric 1
    coverage = compute_word_coverage(gt, pred)

    # build mapping from coverage matches
    mapping = build_speaker_mapping(gt, pred)

    print("Mapping:", mapping)

    # metric 2
    speaker_acc = compute_speaker_accuracy(gt, pred, mapping)

    # final (optional)
    final = coverage * speaker_acc

    print(f"Coverage         : {coverage:.4f}")
    print(f"Speaker Accuracy : {speaker_acc:.4f}")
    print(f"Final Score      : {final:.4f}")