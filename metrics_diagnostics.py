# metrics_and_diagnostics.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional

# Canonical type hint
SecondsSegment = Dict[str, float | str]

# --- CORE HELPER FUNCTIONS (MOVED FROM NOTEBOOK) ---

def segments_to_frames(segments, speakers, frame_len=0.01):
    """
    Convert segment-level annotations into frame-level speaker arrays. 
    (Helper for both GT and SYS).
    """
    assert len(segments) == len(speakers), "Segments and speakers must align"
    
    # Map speakers to IDs
    unique_speakers = list(dict.fromkeys(speakers))
    spk_to_id = {spk: i for i, spk in enumerate(unique_speakers)}

    if not segments:
        return np.array([]), {}

    # Determine max time
    max_time = max(end for _, end in segments) if segments else 0
    T = int(np.ceil(max_time / frame_len))

    frames = np.full(T, None, dtype=object)  # default = silence

    # Fill frames with speaker IDs
    for (start, end), spk in zip(segments, speakers):
        sid = spk_to_id[spk]
        start_idx = int(np.floor(start / frame_len))
        end_idx = int(np.ceil(end / frame_len))
        frames[start_idx:min(end_idx, T)] = sid

    return frames, spk_to_id

def pad_or_trim(arr, target_len, fill=None):
    """Pad or trim a frame array to a target length."""
    cur_len = len(arr)

    if cur_len == target_len:
        return arr

    if cur_len < target_len:
        pad_len = target_len - cur_len
        pad_vals = np.full(pad_len, fill, dtype=object)
        # Use dtype=object to correctly handle mixing None and int IDs
        return np.concatenate([arr, pad_vals])

    return arr[:target_len]

def expand_segments_to_frames(segments, speakers, frame_len=0.01):
    """
    Convert (start, end) segments with speaker labels into frame-level labels.
    (This is used within diagnostics before Hungarian mapping).
    """
    if not segments:
        return np.array([])
        
    T = int(max(e for _, e in segments) / frame_len) + 1
    frames = np.array([None] * T, dtype=object)
    for (s, e), spk in zip(segments, speakers):
        s_idx = int(s / frame_len)
        e_idx = int(e / frame_len)
        frames[s_idx:e_idx] = spk
    return frames

def group_runs(seq):
    """Collapse into (value, run_length) ignoring None."""
    runs = []
    prev, count = None, 0
    for x in seq:
        if x is None:  
            continue
        if x != prev:
            if prev is not None:
                runs.append((prev, count))
            prev, count = x, 1
        else:
            count += 1
    if prev is not None:
        runs.append((prev, count))
    return runs

def extract_frame_errors(gt, sys_remap, frame_len=0.01, window=5, merge_gap=0.5):
    """Extract mismatched segments (GT vs SYS) with local neighborhood."""
    T = len(gt)
    raw = []
    
    # ... (Your original implementation of extract_frame_errors remains here) ...
    # collect mismatched frames + neighborhood
    for i in range(T):
        if gt[i] is None:
            continue
        if sys_remap[i] != gt[i]:
            start = max(0, i - window)
            end = min(T, i + window + 1)
            for j in range(start, end):
                if gt[j] is None:
                    continue
                raw.append((j * frame_len, gt[j], sys_remap[j]))

    if not raw:
        return []

    # sort and merge into intervals (sec-based)
    raw.sort(key=lambda x: x[0])
    start_t, g0, s0 = raw[0]
    end_t = start_t

    errors = []
    for t, g, s in raw[1:]:
        if g == g0 and s == s0 and t - end_t <= merge_gap:
            end_t = t
        else:
            errors.append({
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "gt": g0,
                "sys": s0
            })
            start_t, end_t, g0, s0 = t, t, g, s

    # flush last
    errors.append({
        "start_time": round(start_t, 2),
        "end_time": round(end_t, 2),
        "gt": g0,
        "sys": s0
    })

    return errors


# --- STANDARDIZED DER FUNCTION ---

def der_benchmark_full(
    sys_segments_sec: List[SecondsSegment], # Standardized output of DiarizationPipeline
    meta: Dict[str, Any],
    frame_len: float = 0.01
) -> Tuple[float, float, float, float]:
    """
    Compute DER with decomposition (Miss, FA, Conf). 
    Uses standardized List[SecondsSegment] input.
    """
    # --- 1. GT prep ---
    gt_segments = [(s, e) for s, e in zip(meta["timestamps_start"], meta["timestamps_end"])]
    gt_speakers = meta["speakers"]
    gt_frames, gt_spk_map = segments_to_frames(gt_segments, gt_speakers, frame_len)

    # --- 2. System prep (from List[SecondsSegment]) ---
    sys_segments = [(s["start"], s["end"]) for s in sys_segments_sec]
    sys_labels = [s["speaker"] for s in sys_segments_sec]
    sys_frames, sys_spk_map = segments_to_frames(sys_segments, sys_labels, frame_len)
    
    # --- 3. Align frame length ---
    T = max(len(gt_frames), len(sys_frames))
    gt_frames = pad_or_trim(gt_frames, T)
    sys_frames = pad_or_trim(sys_frames, T)

    n_gt = len(gt_spk_map)
    n_sys = len(sys_spk_map)
    conf_mat = np.zeros((n_gt, n_sys))
    
    # --- 4. Build confusion matrix ---
    for t in range(T):
        g, s = gt_frames[t], sys_frames[t]
        if g is not None and s is not None:
            # g and s are integer IDs due to segments_to_frames output
            conf_mat[g, s] += 1

    # --- 5. Optimal mapping (Hungarian) ---
    if n_gt > 0 and n_sys > 0:
        cost = -conf_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        # mapping: {sys_id: gt_id}
        mapping = {sys: gt for gt, sys in zip(col_ind, row_ind)} # Note: cost shape is (n_gt, n_sys) in original
                                                                # But we map sys_id (col) to gt_id (row)
    else:
        mapping = {}

    # --- 6. Frame-level scoring ---
    missed, fa, conf = 0, 0, 0
    total = sum(1 for g in gt_frames if g is not None)

    for t in range(T):
        g, s = gt_frames[t], sys_frames[t]

        if g is not None and s is None:
            missed += 1
        elif g is None and s is not None:
            fa += 1
        elif g is not None and s is not None:
            mapped_gt = mapping.get(s, None)
            
            # The original logic (if mapped_gt is None) implies that the Hungarian
            # assignment must cover ALL system IDs in the confusion zone. 
            # If s is present in the matrix but not mapped, it is unassigned.
            
            # Simplified check using mapped GT ID:
            if mapped_gt != g:
                conf += 1

    der = (missed + fa + conf) / total if total > 0 else 0.0
    miss_rate = missed / total if total > 0 else 0.0
    fa_rate = fa / total if total > 0 else 0.0
    conf_rate = conf / total if total > 0 else 0.0
    
    print(f"DER={der:.4f} (Miss={miss_rate:.4f}, FA={fa_rate:.4f}, Conf={conf_rate:.4f})")
    
    return der, miss_rate, fa_rate, conf_rate

# --- STANDARDIZED DIAGNOSTICS FUNCTION ---

def label_diagnostics(
    sys_segments_sec: List[SecondsSegment],
    meta: Dict[str, Any], 
    frame_len: float = 0.01
) -> Tuple[Dict[str, Any], str]:
    """Diagnose diarization labelling issues using standardized List[SecondsSegment] input."""

    # --- 1. GT frames (using original label strings) ---
    gt_segments = list(zip(meta["timestamps_start"], meta["timestamps_end"]))
    gt_speakers = meta["speakers"]
    gt = expand_segments_to_frames(gt_segments, gt_speakers, frame_len)

    # --- 2. SYS frames (using original label strings) ---
    sys_segments = [(s["start"], s["end"]) for s in sys_segments_sec]
    sys_labels = [s["speaker"] for s in sys_segments_sec]
    sys = expand_segments_to_frames(sys_segments, sys_labels, frame_len)

    # --- 3. Align lengths ---
    T = max(len(gt), len(sys))
    gt = np.resize(gt, T)
    sys = np.resize(sys, T)

    # --- 4. Hungarian mapping ---
    sys_ids = list(set([s for s in sys if s is not None]))
    gt_ids = list(set([g for g in gt if g is not None]))
    if not sys_ids or not gt_ids:
        return {"mapping": {}, "quality": 0, "issues": [("EMPTY","no labels")]}, "Empty input"

    cost = np.zeros((len(sys_ids), len(gt_ids)))
    for i, s in enumerate(sys_ids):
        for j, g in enumerate(gt_ids):
            # Cost is based on frame overlap (negative sum)
            cost[i, j] = -np.sum((sys == s) & (gt == g))
            
    row_ind, col_ind = linear_sum_assignment(cost)
    # Mapping: {sys_label: gt_label}
    mapping = {sys_ids[i]: gt_ids[j] for i, j in zip(row_ind, col_ind)}

    # Remap sys → gt space
    sys_remap = np.array([mapping.get(s, None) for s in sys])

    # --- 5. Quality, Issues, and Frame Errors ---
    # ... (Your original logic for the rest of the function remains here) ...
    mask = gt != None
    quality = np.mean(sys_remap[mask] == gt[mask]) if np.any(mask) else 0.0

    issues = []
    gt_runs = group_runs(gt)
    sys_runs = group_runs(sys_remap)
    if len(sys_runs) > len(gt_runs):
        issues.append(("OSCILLATION", f"{len(sys_runs)} sys runs vs {len(gt_runs)} gt runs"))

    for s in sys_ids:
        covered = set(gt[sys == s])
        covered.discard(None) # ignore non-speech GT frames
        if len(covered) > 1:
            label = mapping.get(s, f"UNMAPPED_SYS_{s}")
            issues.append(("MERGE", f"sys {label} covers {covered}"))

    for g in gt_ids:
        sys_set = set(sys_remap[gt == g])
        sys_set.discard(None)
        if len(sys_set) > 1:
            label = g
            issues.append(("SPLIT", f"gt {label} split across {sys_set}"))

    frame_errors = extract_frame_errors(gt, sys_remap, frame_len=frame_len)

    diagnostics = {
        "mapping": mapping,
        "quality": float(quality),
        "issues": issues,
        "frame_errors": frame_errors
    }

    summary_parts = [
        f"Quality={quality:.2f}",
        " | ".join([f"{typ}:{msg}" for typ, msg in issues]) if issues else "No major issues"
    ]
    summary = " | ".join(summary_parts)

    return diagnostics, summary