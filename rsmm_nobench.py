# [markdown]
# # Information Retrieval and Summarization on Multiple Users Meeting

# Standard Library
# import os
# import io
# import re
import json
import random
import warnings
from pathlib import Path
from typing import Tuple, List, Dict
from importlib.machinery import SourceFileLoader

# Third-party: Core Scientific / ML
import numpy as np
import torch
import requests
import dotenv

# Hugging Face / Datasets / Transformers
# from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    # Wav2Vec2ProcessorWithLM,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    # pipeline,
)
from transformers.models.wavlm import WavLMModel

# Speech / Audio Models
from whisper.model import Whisper
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference import EncoderClassifier
from groq import Groq

# Audio Processing
import soundfile as sf
from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    VADIterator,
    collect_chunks,
)

# Clustering / Metrics / Dimensionality Reduction
from sklearn.cluster import (
    AgglomerativeClustering,
    SpectralClustering,
    KMeans,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

import umap

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Local Project Utilities
from transform_mp4 import mp4_to_wav

from rsmm_utils import (
    # dump_dataset_paths,
    load_audio,
    # is_titanet,
    ensure_mono,
    pad_or_trim,
    segments_to_frames,
    expand_segments_to_frames,
    group_runs,
    extract_frame_errors,
    normalize_labels,
    save_transcriptions_json,
    concat_json_arrays,
)

from embd_speech_seg import (
    extract_embeddings_from_segments,
    extract_wavlm_embeddings_from_segments,
)

# Warnings Configuration
warnings.filterwarnings("ignore", module="whisper.timing")

def eval_detection(detected, ground_truth, tolerance=0.01):
    """
    Evaluate speech activity detection (VAD) similar to DER
    but only for speech/silence (no speaker labels).

    Parameters
    ----------
    detected : list of (start, end) in seconds
        Predicted speech segments.
    ground_truth : list of (start, end) in seconds
        Reference speech segments.
    tolerance : float, optional
        Extra margin (in seconds) to allow small misalignments.

    Returns
    -------
    recall : float
        Proportion of ground-truth speech time correctly detected.
    fa_rate : float
        Proportion of detected speech time that is false alarm (not in GT).
    """

    # Flatten into frame-level timeline with step resolution
    # (here 10ms frames, similar to pyannote)
    frame_hop = 0.01
    if not ground_truth or not detected:
        return 0.0, 0.0

    # Find total duration
    max_time = max(max(e for _, e in ground_truth), max(e for _, e in detected))

    n_frames = int(max_time / frame_hop) + 1
    gt_activity = [0] * n_frames
    det_activity = [0] * n_frames

    # Mark GT frames
    for s, e in ground_truth:
        s_i = int((s - tolerance) / frame_hop)
        e_i = int((e + tolerance) / frame_hop)
        for i in range(max(0, s_i), min(n_frames, e_i)):
            gt_activity[i] = 1

    # Mark detected frames
    for s, e in detected:
        s_i = int((s - tolerance) / frame_hop)
        e_i = int((e + tolerance) / frame_hop)
        for i in range(max(0, s_i), min(n_frames, e_i)):
            det_activity[i] = 1

    # Compute overlaps
    tp = sum(1 for g, d in zip(gt_activity, det_activity) if g == 1 and d == 1)
    fn = sum(1 for g, d in zip(gt_activity, det_activity) if g == 1 and d == 0)
    fp = sum(1 for g, d in zip(gt_activity, det_activity) if g == 0 and d == 1)

    print("True Pos, False Neg, False Pos", tp, fn, fp)
    recall = tp / (tp + fn + 1e-9)
    fa_rate = fp / (tp + fp + 1e-9)

    return recall, fa_rate


# [markdown]
# ### Clustering


def spectral_cluster_eigengap(embeddings, min_clusters=2, max_clusters=20):
    """
    Spectral clustering with cluster count determined by the eigen-gap heuristic.

    Parameters
    ----------
    embeddings : array-like, shape (n_samples, n_features)
        Speaker embeddings.
    min_clusters : int, default=2
        Minimum number of clusters to consider.
    max_clusters : int, default=20
        Maximum number of clusters to consider.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels assigned to each embedding.
    k_opt : int
        Estimated optimal number of clusters (from eigen-gap).
    eigvals : ndarray
        Sorted eigenvalues of the Laplacian.
    gaps : ndarray
        Differences between consecutive eigenvalues.
    """

    # Step 1: similarity (cosine)
    sim = cosine_similarity(embeddings)

    # Step 2: degree matrix
    degrees = sim.sum(axis=1)
    D = np.diag(degrees)

    # Step 3: normalized Laplacian
    with np.errstate(divide="ignore"):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    L = np.eye(len(sim)) - D_inv_sqrt @ sim @ D_inv_sqrt

    # Step 4: eigen decomposition
    eigvals, _ = eigh(L)
    eigvals = np.sort(eigvals)[: max_clusters + 1]

    # Step 5: eigen-gap
    gaps = np.diff(eigvals)
    search_region = gaps[min_clusters - 1 : max_clusters - 1]
    best_idx = np.argmax(search_region)
    k_opt = best_idx + min_clusters

    # Step 6: spectral clustering with k_opt
    clustering = SpectralClustering(
        n_clusters=k_opt,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=0,
    ).fit(embeddings)

    return clustering.labels_, k_opt, eigvals, gaps


def agglomerative_k_search(
    embeddings: np.ndarray,
    metric: str = "cosine",
    linkage_method: str = "average",
    min_clusters: int = 2,
    max_clusters: int = 20,
):
    """
    Cluster embeddings with Agglomerative (dendrogram once, cut by k).
    Picks best k in [min_clusters, max_clusters] using silhouette score.

    Returns:
        best_labels: np.ndarray of shape (N,)
        best_k: int, chosen number of clusters
        best_score: float
        Z: linkage matrix
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), 1, 0.0, None

    # Step 1: compute pairwise distances and dendrogram (once)
    condensed = pdist(embeddings, metric=metric)
    Z = linkage(condensed, method=linkage_method)

    # Step 2: cache full distance matrix for silhouette
    dist_mat = squareform(condensed)

    best_score = -np.inf
    best_labels, best_k = None, None

    # Step 3: sweep possible k values
    for k in range(min_clusters, min(max_clusters, n_samples - 1) + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_clusters = len(set(labels))
        if n_clusters < 2:
            continue

        try:
            score = silhouette_score(dist_mat, labels, metric="precomputed")
        except Exception:
            score = -davies_bouldin_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = n_clusters

    # Fallback: if nothing valid, just cut at min_clusters
    if best_labels is None:
        best_labels = fcluster(Z, t=min_clusters, criterion="maxclust")
        best_k = min_clusters
        try:
            best_score = silhouette_score(dist_mat, best_labels, metric="precomputed")
        except Exception:
            best_score = -davies_bouldin_score(embeddings, best_labels)

    return best_labels, best_k, best_score


def umap_cluster_embeddings(
    embeddings,
    reducer_dim=10,
    gt_labels=None,
    n_clusters=None,
    random_state=42,
):
    """
    UMAP -> KMeans clustering.

    Returns:
      speaker_labels (list[str])   - e.g. ["SPEAKER_00", ...] (compatible with your pipeline)
      cluster_ids (ndarray[int])   - raw numeric cluster ids from KMeans
      emb_reduced (ndarray)        - UMAP-reduced embeddings (n x reducer_dim)
      label2id (dict)              - map from string label -> numeric id (for plotting)
      results (dict)               - metrics and extra info
    """
    # 1) UMAP reduction
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=reducer_dim,
        metric="cosine",
        random_state=random_state,
    )
    emb_reduced = reducer.fit_transform(embeddings)

    # print(len(set(gt_labels)))
    # 2) cluster count
    if n_clusters is None:
        if gt_labels is not None:
            n_clusters = len(set(gt_labels))
        else:
            raise ValueError("Need n_clusters or gt_labels to infer cluster count")

    # 3) clustering
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_ids = km.fit_predict(
        emb_reduced
    )  # numeric 0..K-1 (but order not guaranteed)

    # 4) build stable id->label using order-of-appearance of cluster ids
    unique_clusters = list(dict.fromkeys(cluster_ids))  # preserve appearance order
    id2label = {cid: f"SPEAKER_{i:02d}" for i, cid in enumerate(unique_clusters)}
    speaker_labels = [id2label[cid] for cid in cluster_ids]

    # 5) label -> numeric id mapping (for plotting)
    unique_labels = [id2label[cid] for cid in unique_clusters]
    label2id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label2id[lbl] for lbl in speaker_labels], dtype=int)

    # 6) metrics (use numeric cluster_ids for ARI/NMI)
    results = {
        "umap_embeddings": emb_reduced,
        "cluster_ids": cluster_ids,
        "label2id": label2id,
        "numeric_labels": numeric_labels,
        "ari": None,
        "nmi": None,
    }
    if gt_labels is not None and len(gt_labels) == len(cluster_ids):
        results["ari"] = adjusted_rand_score(gt_labels, cluster_ids)
        results["nmi"] = normalized_mutual_info_score(gt_labels, cluster_ids)

    return speaker_labels, cluster_ids, emb_reduced, label2id, results


def transcribe_chunks_ctc(waveform, sr, chunks, model, processor, device="cuda"):
    """
    Vietnamese Wav2Vec2 CTC chunk-based transcription.
    - waveform: torch.Tensor [1, N]
    - chunks: list of {start, end}
    """
    model = model.to(device)
    transcripts = []

    for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="Transcribing"):
        # Slice chunk (samples)
        segment = waveform[:, c["start"] : c["end"]].cpu().numpy().squeeze()

        # Preprocess
        inputs = processor(
            segment, sampling_rate=sr, return_tensors="pt", padding="longest"
        )

        input_values = inputs.input_values.to(device)
        attention_mask = (
            inputs.attention_mask.to(device) if "attention_mask" in inputs else None
        )

        # Forward CTC
        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode
        text = processor.batch_decode(predicted_ids)[0]

        transcripts.append(
            {
                "chunk_id": i,
                "start": c["start"],
                "end": c["end"],
                "text": text.strip(),
                "words": None,  # CTC does not produce word timestamps
            }
        )

    return transcripts


def transcribe_chunks_hf_batch(waveform, sr, chunks, model, processor, batch_size=1):
    device = next(model.parameters()).device
    transcripts = []

    batch_segments = []
    batch_meta = []

    def flush_batch():
        if len(batch_segments) == 0:
            return []

        # Preprocess everything in the batch
        inputs = processor(
            batch_segments, sampling_rate=sr, return_tensors="pt", padding=True
        ).to(device)

        input_features = inputs["input_features"].to(device)
        forced_ids = processor.get_decoder_prompt_ids(
            language="vi",
            task="transcribe",
        )
        with torch.no_grad():
            generated_ids = model.generate(
                input_features, forced_decoder_ids=forced_ids, max_length=448
            )

        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Attach metadata
        out = []
        for meta, text in zip(batch_meta, texts):
            out.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "start": meta["start"],
                    "end": meta["end"],
                    "text": text.strip(),
                    "words": None,
                }
            )

        return out

    # ---- batching loop ----
    for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="Transcribing"):
        segment = (
            waveform[:, c["start"] : c["end"]].cpu().numpy().squeeze().astype("float32")
        )

        batch_segments.append(segment)
        batch_meta.append({"chunk_id": i, "start": c["start"], "end": c["end"]})

        if len(batch_segments) >= batch_size:
            transcripts.extend(flush_batch())
            batch_segments.clear()
            batch_meta.clear()

    # Flush remaining
    transcripts.extend(flush_batch())

    # Sort by chunk_id (preserve deterministic order)
    transcripts.sort(key=lambda x: x["chunk_id"])
    return transcripts


def transcribe_chunks(waveform, sr, chunks, asr_model: Whisper):
    """
    Run Whisper ASR on pre-chunked waveform segments.
    - waveform: torch.Tensor [1, num_samples]
    - sr: sample rate
    - chunks: list of dicts with {start, end, segments} in samples
    - asr_model: loaded Whisper model
    Returns transcripts aligned in **samples** (not seconds).
    """
    transcripts = []

    for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="Transcribing"):
        # Slice chunk by samples
        segment = waveform[:, c["start"] : c["end"]].cpu().numpy()
        segment = segment.squeeze().astype("float32")

        # Normalize if needed
        # if segment.max() > 1.0:
        #     segment = segment / max(1e-9, abs(segment).max())

        # Run Whisper with timestamps
        result = asr_model.transcribe(
            segment,
            fp16=False,  # safer on CPU/small GPU
            word_timestamps=True,  # return per-word timestamps
            beam_size=3,  # beam search for stability
            temperature=(0, 0.1, 0.2, 0.4, 0.8),  # deterministic output
            compression_ratio_threshold=1.8,
        )

        # Convert Whisper word timestamps (sec) → samples
        words = []
        for w in result.get("segments", []):
            for item in w["words"]:
                words.append(
                    {
                        "word": item["word"],
                        "start": c["start"] + int(item["start"] * sr),
                        "end": c["start"] + int(item["end"] * sr),
                    }
                )

        transcripts.append(
            {
                "chunk_id": i,
                "start": c["start"],  # in samples
                "end": c["end"],  # in samples
                "text": result["text"].strip(),
                "words": words,
            }
        )

        # Optional debug log in seconds
        # print(f"[Chunk {i}] {c['start']/sr:.2f}-{c['end']/sr:.2f}s: {result['text'].strip()}")
        # print(f"[Chunk {i}] {c['start']/sr:.4f}-{c['end']/sr:.4f}s", end=" ")

    return transcripts


def chunk_by_silence_and_overlap(
    speech_timestamps,
    sr,
    min_silence=2.0,
    max_chunk=60.0,
    overlap=2.0,
):
    """
    Create ASR chunks from diarization speech timestamps.
    Works in samples, not seconds.
    Ensures that overlapping windows only carry their relevant segments.
    """
    min_silence_samples = int(min_silence * sr)
    max_chunk_samples = int(max_chunk * sr)
    overlap_samples = int(overlap * sr)

    chunks = []
    cur_segments = []
    cur_start, cur_end = None, None

    # --- First pass: group by silence gaps ---
    for seg in speech_timestamps:
        start, end = (seg["start"], seg["end"]) if isinstance(seg, dict) else seg

        if cur_start is None:
            cur_start, cur_end = start, end
            cur_segments = [(start, end)]
            continue

        gap = start - cur_end

        if gap >= min_silence_samples:
            chunks.append(
                {"start": cur_start, "end": cur_end, "segments": cur_segments}
            )
            cur_start, cur_end = start, end
            cur_segments = [(start, end)]
        else:
            cur_end = end
            cur_segments.append((start, end))

    if cur_segments:
        chunks.append({"start": cur_start, "end": cur_end, "segments": cur_segments})

    # --- Second pass: split long chunks with overlap ---
    final_chunks = []
    for c in chunks:
        duration = c["end"] - c["start"]
        if duration <= max_chunk_samples:
            final_chunks.append(c)
        else:
            start = c["start"]
            while start < c["end"]:
                end = min(start + max_chunk_samples, c["end"])

                # ✅ keep only segments that intersect this sub-window
                sub_segments = [
                    (s, e) for (s, e) in c["segments"] if e > start and s < end
                ]

                final_chunks.append(
                    {"start": start, "end": end, "segments": sub_segments}
                )

                if end == c["end"]:
                    break
                start = end - overlap_samples  # slide with overlap

    return final_chunks


def assign_speakers(transcripts, diar_segments, sr, snap_gap_sec=1):
    snap_gap_samples = int(snap_gap_sec * sr)
    results = []
    cur_speaker, cur_words = None, []

    # flatten word-level info across chunks
    words_all = []
    for t in transcripts:
        words_all.extend(t["words"])

    for w in words_all:
        w_mid = (w["start"] + w["end"]) // 2

        # find diar segment covering this word
        candidates = [
            seg for seg in diar_segments if seg["start"] <= w_mid <= seg["end"]
        ]
        if candidates:
            speaker = candidates[0]["speaker"]
        else:
            # nearest diar segment
            nearest = min(
                diar_segments,
                key=lambda s: min(abs(w_mid - s["start"]), abs(w_mid - s["end"])),
            )
            gap = min(abs(w_mid - nearest["start"]), abs(w_mid - nearest["end"]))
            speaker = nearest["speaker"] if gap <= snap_gap_samples else None

        # group by speaker
        if speaker != cur_speaker:
            if cur_words:
                results.append(
                    {
                        "speaker": (
                            f"Speaker {cur_speaker}"
                            if isinstance(cur_speaker, (int, np.integer))
                            else (cur_speaker or "Unknown")
                        ),
                        "start": min(wd["start"] for wd in cur_words),
                        "end": max(wd["end"] for wd in cur_words),
                        "text": " ".join(wd["word"] for wd in cur_words),
                    }
                )
            cur_speaker, cur_words = speaker, [w]
        else:
            cur_words.append(w)

    # flush last
    if cur_words:
        results.append(
            {
                "speaker": (
                    f"Speaker {cur_speaker}"
                    if isinstance(cur_speaker, (int, np.integer))
                    else (cur_speaker or "Unknown")
                ),
                "start": min(wd["start"] for wd in cur_words),
                "end": max(wd["end"] for wd in cur_words),
                "text": " ".join(wd["word"] for wd in cur_words),
            }
        )

    return results

def run_vad_segmentation(waveform, vad_model, sample_rate):
    return get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.1,
        max_speech_duration_s=5.0,
        min_speech_duration_ms=50,
        min_silence_duration_ms=50,
    )


def split_segments_by_length(speech_segments, sample_rate, min_len_sec=0.6):
    min_len = int(min_len_sec * sample_rate)
    valid, skipped = [], []
    for ts in speech_segments:
        if ts["end"] - ts["start"] < min_len:
            skipped.append(ts)
        else:
            valid.append(ts)
    return valid, skipped


def extract_large_embeddings(waveform, encoder, segments, seg_len_sec=2.0):
    return (
        extract_embeddings_from_segments(
            waveform,
            encoder,
            segments,
            n_samples=3,
            seg_len_sec=seg_len_sec,
            skip_short=False,
        )
        .squeeze(1)
        .numpy()
    )


def normalize_if_needed(x, use_cosine_norm):
    if not use_cosine_norm:
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n


def cluster_large_segments(valid_segments, embeddings_norm):
    speaker_labels, _, _, _ = spectral_cluster_eigengap(embeddings_norm)
    labeled = [
        {"speaker": f"Speaker {lbl}", "start": seg["start"], "end": seg["end"]}
        for seg, lbl in zip(valid_segments, speaker_labels)
    ]
    return speaker_labels, len(set(speaker_labels)), labeled


def assign_segments_to_clusters(embeddings, centers, metric="cosine"):
    if metric == "cosine":
        norms_emb = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        norms_ctr = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
        sim = (embeddings @ centers.T) / (norms_emb * norms_ctr.T)
        labels = np.argmax(sim, axis=1)
        conf = np.max(sim, axis=1)
    else:
        # Placeholder: you can plug Mahalanobis, PLDA, or density-based logic here
        dists = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=-1)
        labels = np.argmin(dists, axis=1)
        conf = 1 / (1 + np.min(dists, axis=1))
    return labels, conf


def dbg_print(msg):
    print("[DBG]", msg)


def check_parent_splitting(valid_large_segments, phase2_result, merged_segments):
    # For each parent, if any sub in phase2_result["outliers"] falls inside parent,
    # check merged_segments contains splits (parent is not present intact covering outlier)
    outliers = phase2_result.get("outliers", [])
    intact_parents = []
    for parent in valid_large_segments:
        p_s, p_e = parent["start"], parent["end"]
        parent_present = any(
            m["start"] <= p_s + 1e-6 and m["end"] >= p_e - 1e-6 for m in merged_segments
        )
        # if there are outliers inside parent
        outs_in_parent = [o for o in outliers if o["start"] >= p_s and o["end"] <= p_e]
        if outs_in_parent and parent_present:
            intact_parents.append(parent)
    dbg_print(
        f"Parents containing outliers but still present intact in merged_segments: {len(intact_parents)}"
    )
    if intact_parents:
        dbg_print("Example intact parents: " + str(intact_parents[:3]))


def refine_clusters_with_wavlm(
    waveform,
    encoder_small,
    valid_segments,
    skipped_segments,
    speaker_labels,
    n_speakers,
    use_cosine_norm,
    seg_len_small=0.3,
    stride_small=0.1,
):
    """
    Phase 2 — Intra-cluster refinement using WavLM.
    Unified output keys for consistent downstream use.
    """
    sample_rate = 16000
    seg_samples = int(seg_len_small * sample_rate)
    stride_samples = int(stride_small * sample_rate)

    # Edge case: no valid segments
    if not valid_segments:
        return {
            "refined_clusters": {},
            "segments_refined": [],
            "outliers": [],
            "outlier_embeddings": np.empty((0, 0)),
            "skipped_embeddings": np.empty((0, 0)),
            "skipped_segments": [],
            "all_embeddings": np.empty((0, 0)),
            "all_labels": [],
        }

    # ------------------------------
    # Step 1 — build subsegments
    # ------------------------------
    all_subsegments, all_labels = [], []
    for seg, lbl in zip(valid_segments, speaker_labels):
        start, end = seg["start"], seg["end"]
        seg_dur = end - start
        if seg_dur < seg_samples:
            continue

        # stride = min(int(seg_samples * 0.5), stride_samples)
        max_subs = 20
        positions = list(range(start, end - seg_samples + 1, stride_samples))
        if len(positions) > max_subs:
            step = max(1, len(positions) // max_subs)
            positions = positions[::step]

        for s in positions:
            all_subsegments.append({"start": s, "end": s + seg_samples})
            all_labels.append(lbl)

    if not all_subsegments:
        return {
            "refined_clusters": {},
            "segments_refined": [],
            "outliers": [],
            "outlier_embeddings": np.empty((0, 0)),
            "skipped_embeddings": np.empty((0, 0)),
            "skipped_segments": skipped_segments,
            "all_embeddings": np.empty((0, 0)),
            "all_labels": [],
        }

    # ------------------------------
    # Step 2 — embed
    # ------------------------------
    all_embs = (
        extract_wavlm_embeddings_from_segments(
            waveform,
            encoder_small,
            all_subsegments,
            seg_len_sec=seg_len_small,
            n_samples=1,
            skip_short=False,
        )
        .squeeze(1)
        .numpy()
    )
    emb_dim = all_embs.shape[1]
    all_embs = normalize_if_needed(all_embs, True)
    cluster_subembs = {i: [] for i in range(n_speakers)}
    cluster_segments = {i: [] for i in range(n_speakers)}
    for emb, seg, lbl in zip(all_embs, all_subsegments, all_labels):
        cluster_subembs[lbl].append(emb)
        cluster_segments[lbl].append(seg)
    for lbl in range(n_speakers):
        cluster_subembs[lbl] = (
            np.stack(cluster_subembs[lbl])
            if cluster_subembs[lbl]
            else np.empty((0, emb_dim))
        )
    # Global spectral clustering on all subsegment embeddings and comparison with original labels
    # --- Global clustering ---
    all_embs_norm_global = normalize_if_needed(all_embs, use_cosine_norm)

    # run spectral clustering (same helper as before)
    global_labels, _, _, _ = spectral_cluster_eigengap(all_embs_norm_global)
    global_labels = np.asarray(global_labels)

    # map original string labels to ints (preserve first-seen order)
    unique_true = list(dict.fromkeys(all_labels))
    true_map = {lab: i for i, lab in enumerate(unique_true)}
    true_ids = np.array([true_map[l] for l in all_labels], dtype=int)

    # --- Build contingency matrix (pred_clusters x true_labels) ---
    pred_cluster_ids = np.unique(global_labels)
    n_pred = pred_cluster_ids.max() + 1
    n_true = len(unique_true)
    cont = np.zeros((n_pred, n_true), dtype=int)

    for p, t in zip(global_labels, true_ids):
        cont[int(p), int(t)] += 1

    # --- Best one-to-one assignment (Hungarian algorithm) ---
    m = max(n_pred, n_true)
    cost = np.zeros((m, m), dtype=int)
    cost[:n_pred, :n_true] = -cont  # negative because linear_sum_assignment minimizes
    row_ind, col_ind = linear_sum_assignment(cost)

    # --- Evaluate accuracy + mapping ---
    matched = 0
    label_map = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_pred and c < n_true:
            matched += cont[r, c]
            label_map[r] = c  # map predicted cluster r → true label c

    accuracy = matched / len(all_labels)

    # --- Optional: remap predicted labels to true label indices ---
    mapped_labels = np.array([label_map.get(lbl, lbl) for lbl in global_labels])
    # you can then use mapped_labels instead of raw global_labels for evaluation or saving

    # --- Summary for inspection ---
    global_clustering_comparison = {
        "n_pred_clusters": int(n_pred),
        "n_true_labels": int(n_true),
        "true_label_names": unique_true,
        "contingency": cont,
        "label_map": label_map,  # predicted cluster -> true label index
        "accuracy": float(accuracy),
    }

    print(
        f"Global spectral clustering: {n_pred} clusters vs {n_true} true labels, accuracy={accuracy:.4f}"
    )
    print("Label map (pred→true):", label_map)
    # ------------------------------
    # Step 3 — refinement
    # ------------------------------
    refined_clusters = {}
    all_outliers, all_outlier_embs, refined_segments = [], [], []

    for i in range(n_speakers):
        subembs = cluster_subembs[i]
        if len(subembs) == 0:
            continue
        subembs_norm = normalize_if_needed(subembs, use_cosine_norm)
        sub_labels, _, _, _ = spectral_cluster_eigengap(subembs_norm)

        subclusters = {}
        local_outliers, local_outlier_embs = [], []

        for lbl in np.unique(sub_labels):
            sub_idx = np.where(sub_labels == lbl)[0]
            sub_embs = subembs_norm[sub_idx]
            centroid = np.mean(sub_embs, axis=0)

            dists = np.linalg.norm(sub_embs - centroid, axis=1)
            thresh = np.mean(dists) + 1.5 * np.std(dists)
            outlier_mask = dists > thresh

            outlier_idx = sub_idx[outlier_mask]
            kept_idx = sub_idx[~outlier_mask]

            outlier_embs = sub_embs[outlier_mask]
            outlier_segments = [cluster_segments[i][j] for j in outlier_idx]
            kept_segments = [cluster_segments[i][j] for j in kept_idx]

            local_outliers.extend(outlier_segments)
            if len(outlier_embs) > 0:
                local_outlier_embs.append(outlier_embs)

            # Save subcluster
            subclusters[int(lbl)] = {
                "embs": sub_embs[~outlier_mask],
                "segments": kept_segments,
                "centroid": centroid,
                "count": len(kept_segments),
            }

            # Flat refined segments
            for seg in kept_segments:
                seg = {**seg, "speaker": f"Speaker{i}"}
                refined_segments.append(seg)

        if local_outlier_embs:
            local_outlier_embs = np.concatenate(local_outlier_embs, axis=0)
            all_outlier_embs.append(local_outlier_embs)

        refined_clusters[i] = {
            "subembs": subembs_norm,
            "sublabels": sub_labels,
            "subclusters": subclusters,
            "outliers": local_outliers,
        }

        all_outliers.extend(local_outliers)

    if all_outlier_embs:
        all_outlier_embs = np.concatenate(all_outlier_embs, axis=0)
    else:
        all_outlier_embs = np.empty((0, emb_dim))

    # ------------------------------
    # Step 4 — skipped segments
    # ------------------------------
    if skipped_segments:
        skipped_embs = (
            extract_wavlm_embeddings_from_segments(
                waveform,
                encoder_small,
                skipped_segments,
                seg_len_sec=seg_len_small,
                n_samples=1,
                skip_short=False,
            )
            .squeeze(1)
            .numpy()
        )
    else:
        skipped_embs = np.empty((0, emb_dim))
    skipped_embs_norm = normalize_if_needed(skipped_embs, True)
    assert (
        all_embs.shape[1] == skipped_embs.shape[1]
    ), f"outliers={all_embs.shape[1]}, skipped={skipped_embs.shape[1]}"
    return {
        "refined_clusters": refined_clusters,
        "refined_segments": refined_segments,
        "outliers": all_outliers,
        "outlier_embeddings": all_outlier_embs,
        "skipped_embeddings": skipped_embs_norm,
        "skipped_segments": skipped_segments,
        "all_embeddings": all_embs,
        "all_labels": all_labels,
    }


def relabel_outliers_and_skipped(
    refined_clusters,
    outlier_embeddings=None,
    outlier_segments=None,
    skipped_embeddings=None,
    skipped_segments=None,
    assign_threshold=0.35,
    use_cosine_norm=False,
):
    """
    Phase 3: assign outlier + skipped candidates to refined subcluster centroids.

    Inputs:
      - refined_clusters: dict from Phase2. Each value has "subclusters": {subid: {"centroid","embs","segments",...}, ...}
      - outlier_embeddings: np.ndarray (N_out, D) or None
      - outlier_segments: list of segment dicts (N_out) or None
      - skipped_embeddings: np.ndarray (N_skip, D) or None
      - skipped_segments: list of segment dicts (N_skip) or None

    Behavior:
      - Merge outliers + skipped into one candidate list (outliers first).
      - Use subcluster centroids as targets.
      - Compute cosine similarity and assign each candidate to nearest centroid.
      - If sim >= assign_threshold => "high" confidence, else "tentative" (still assigned to nearest).
      - Always returns labels for every candidate (zero miss-rate).
    Returns:
      {
        "segments_reassigned": [ {segment_dict + speaker, confidence, tentative, assigned_to}, ... ],
        "assignments": [speaker_label_str, ...],
        "n_speakers_total": total_count (coarse clusters count),
        "centroid_keys": centroid_key_list  # optional, useful for debugging
      }
    """

    # Normalize inputs / defaults
    outlier_embeddings = (
        np.asarray(outlier_embeddings)
        if outlier_embeddings is not None
        else np.empty((0, 0))
    )
    skipped_embeddings = (
        np.asarray(skipped_embeddings)
        if skipped_embeddings is not None
        else np.empty((0, 0))
    )
    outlier_segments = list(outlier_segments) if outlier_segments is not None else []
    skipped_segments = list(skipped_segments) if skipped_segments is not None else []

    # Build candidate embeddings and segments in stable order
    cand_embs_list = []
    cand_segs = []

    if outlier_embeddings.size != 0:
        cand_embs_list.append(outlier_embeddings)
        cand_segs.extend(outlier_segments)
    if skipped_embeddings.size != 0:
        cand_embs_list.append(skipped_embeddings)
        cand_segs.extend(skipped_segments)

    if len(cand_segs) == 0:
        return {
            "segments_reassigned": [],
            "assignments": [],
            "n_speakers_total": len(refined_clusters),
            "centroid_keys": [],
        }

    # stack embeddings (handle case where either array is empty)
    cand_embs = np.vstack(cand_embs_list) if len(cand_embs_list) > 0 else np.empty((0,))

    # Build list of centroids and keys (coarse_id, subid)
    centroids = []
    centroid_keys = []
    for coarse_id, info in refined_clusters.items():
        subcls = info.get("subclusters", {})
        for subid, sinfo in subcls.items():
            cent = sinfo.get("centroid", None)
            if cent is None:
                embs = np.asarray(sinfo.get("embs", []))
                if embs.size == 0:
                    continue
                cent = np.mean(embs, axis=0)
            centroids.append(np.asarray(cent))
            centroid_keys.append((coarse_id, subid))

    if len(centroids) == 0:
        # No centroids: mark all as new tentative speakers
        labeled = []
        next_new_id = max(refined_clusters.keys(), default=-1) + 1
        for seg in cand_segs:
            lbl = f"Speaker{next_new_id}"
            labeled.append(
                {
                    **seg,
                    "speaker": lbl,
                    "confidence": 0.0,
                    "tentative": True,
                    "assigned_to": None,
                }
            )
            next_new_id += 1
        return {
            "segments_reassigned": labeled,
            "assignments": [s["speaker"] for s in labeled],
            "n_speakers_total": next_new_id,
            "centroid_keys": [],
        }

    centroids = np.stack(centroids)  # (M, D)

    # Normalize if using cosine
    if use_cosine_norm:
        # normalize centroids and candidates safely
        cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
        centroids_normed = centroids / cent_norms
        cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9
        cand_embs_normed = cand_embs / cand_norms
    else:
        centroids_normed = centroids
        cand_embs_normed = cand_embs

    # cosine similarity matrix (N_candidates x M_centroids)
    sims = cand_embs_normed.dot(centroids_normed.T)

    best_idx = np.argmax(sims, axis=1)
    best_val = np.max(sims, axis=1).astype(float)

    # Assign to nearest centroid, mark tentative if below threshold
    labeled_segments = []
    assignments = []
    for i, seg in enumerate(cand_segs):
        sim = float(best_val[i])
        centroid_index = int(best_idx[i])
        coarse_id, subid = centroid_keys[centroid_index]

        if sim >= assign_threshold:
            label = f"Speaker{coarse_id}"
            tentative = False
        else:
            # still assign nearest, but mark tentative to allow later re-eval
            label = f"Speaker{coarse_id}"
            tentative = True

        assigned_info = {
            **seg,
            "speaker": label,
            "confidence": sim,
            "tentative": tentative,
            "assigned_to": {"coarse": coarse_id, "sub": subid},
        }
        labeled_segments.append(assigned_info)
        assignments.append(label)

    n_total = len(cand_segs)
    n_tent = sum(1 for s in labeled_segments if s["tentative"])
    avg_conf = np.mean(best_val) if len(best_val) else 0
    min_conf = np.min(best_val) if len(best_val) else 0
    max_conf = np.max(best_val) if len(best_val) else 0

    print(
        f"[DBG] Reassignment: total={n_total}, tentative={n_tent}, "
        f"avg_conf={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}"
    )
    print(
        "[DBG] Similarity distribution:",
        np.histogram(best_val, bins=np.linspace(0, 1, 11))[0],
    )

    n_speakers_total = max(refined_clusters.keys(), default=-1) + 1

    return {
        "segments_reassigned": labeled_segments,
        "assignments": assignments,
        "n_speakers_total": n_speakers_total,
        "centroid_keys": centroid_keys,
    }


def merge_segments_consistently(
    labeled_large_segments,
    refined_segments,
    relabeled_segments,
    tolerance=0.05,  # small time tolerance in seconds
):
    """
    Combine segments from all stages into one consistent, non-overlapping list.
    Priority:
        1. relabeled_segments (outliers + skipped reassigned)
        2. refined_segments (WavLM small segments)
        3. labeled_large_segments (ECAPA base)
    """

    # Combine all with priority tagging
    all_segments = []
    for seg in labeled_large_segments:
        all_segments.append({**seg, "priority": 0})
    for seg in refined_segments:
        all_segments.append({**seg, "priority": 1})
    for seg in relabeled_segments:
        all_segments.append({**seg, "priority": 2})

    # Sort by start time, then by priority descending
    all_segments.sort(key=lambda s: (s["start"], -s["priority"]))

    merged = []
    for seg in all_segments:
        s, e = seg["start"], seg["end"]
        # Skip if fully covered by higher-priority segment
        overlapped = False
        for m in merged:
            if s >= m["start"] - tolerance and e <= m["end"] + tolerance:
                overlapped = True
                break
        if overlapped:
            continue

        # If overlaps partially, trim or split intelligently
        trimmed_start, trimmed_end = s, e
        for m in merged:
            if not (e <= m["start"] or s >= m["end"]):
                if s < m["start"] and e > m["start"]:
                    trimmed_end = min(e, m["start"])
                elif e > m["end"] and s < m["end"]:
                    trimmed_start = max(s, m["end"])
        if trimmed_end - trimmed_start > 0:
            merged.append({**seg, "start": trimmed_start, "end": trimmed_end})

    # Sort by start again to ensure order
    merged.sort(key=lambda x: x["start"])
    return merged


def visualize_embeddings(emb_large, emb_small, labels_large, labels_small):
    try:
        reducer = umap.UMAP(
            n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42
        )
    except Exception:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)

    emb_large_2d = reducer.fit_transform(emb_large)
    emb_small_2d = reducer.transform(emb_small)
    print(
        "set size and items",
        len(set(labels_large)),
        len(set(labels_small)),
        set(labels_large),
        set(labels_small),
    )
    norm_labels_large = normalize_labels(labels_large)
    norm_labels_small = normalize_labels(labels_small)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(
        emb_large_2d[:, 0], emb_large_2d[:, 1], c=norm_labels_large, cmap="tab10", s=30
    )
    plt.title("Phase 1 — Large embeddings (coarse clusters)")

    plt.subplot(1, 2, 2)
    plt.scatter(
        emb_small_2d[:, 0],
        emb_small_2d[:, 1],
        c=norm_labels_small,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )
    plt.title("Phase 2 — Skipped short segments (fine clustering)")
    plt.tight_layout()
    plt.show()


def diarization(
    waveform,
    sample_rate,
    vad_model,
    encoder_large,
    encoder_small,
    meta=None,
    use_cosine_norm: bool = False,
    plot: bool = True,
):
    """
    Two-phase diarization:
      - Phase 1: coarse clusters from large encoder (ECAPA/Titanet)
      - Phase 2: fine clusters from small encoder (WavLM), on skipped short segments
    """

    # --- Phase 1: large segments and ECAPA embeddings ---
    speech_segments = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.1,
        # neg_threshold=0.2,
        max_speech_duration_s=3.0,
        min_speech_duration_ms=100,
        min_silence_duration_ms=50,
    )

    seg_len_large = 1.2
    seg_len_small = 0.5

    # 🔸 capture which segments were skipped due to length
    valid_segments = []
    skipped_segments = []

    # seg_len = int(seg_len_large * sample_rate)
    min_len = int(0.5 * sample_rate)  # same as min_len_ratio * sample_rate

    # partition segments before extraction
    for ts in speech_segments:
        start, end = ts["start"], ts["end"]
        seg_len_actual = end - start
        if seg_len_actual < min_len:
            skipped_segments.append(ts)
        else:
            valid_segments.append(ts)

    embeddings_large = (
        extract_wavlm_embeddings_from_segments(
            waveform,
            encoder_small,
            valid_segments,  # only long enough ones
            n_samples=3,
            seg_len_sec=seg_len_large,
            skip_short=False,  # we've already filtered manually
        )
        .squeeze(1)
        .numpy()
    )

    if use_cosine_norm:

        def normalize_rows(x):
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
            return x / n

        embeddings_large_norm = normalize_rows(embeddings_large)
    else:
        embeddings_large_norm = embeddings_large

    # --- Phase 1 clustering ---
    speaker_labels, _, _, _ = spectral_cluster_eigengap(
        embeddings_large_norm, min_clusters=3, max_clusters=10
    )
    labeled_large_segments = [
        {"speaker": f"Speaker {lbl}", "start": seg["start"], "end": seg["end"]}
        for seg, lbl in zip(valid_segments, speaker_labels)
    ]
    n_speakers = len(set(speaker_labels))
    print("Num. of Speakers:", n_speakers)

    # --- Phase 2: fine clustering on skipped short segments ---
    # 🔸 reuse skipped_segments instead of new VAD pass
    if len(skipped_segments) == 0:
        print("No skipped short segments — skipping small phase.")
        return {
            "segments_large": labeled_large_segments,
            "segments_small": [],
            "merged_segments": labeled_large_segments,
            "labels": [seg["speaker"] for seg in labeled_large_segments],
            "pivots": None,
        }

        # --- Leak pivots: convert large-cluster representatives into WavLM-space ---
    # Initialize a dict (or list) of pivot segments per cluster
    pivot_segments_by_cluster = {i: [] for i in range(n_speakers)}

    for i in range(n_speakers):
        cluster_segments = [
            seg for seg, lbl in zip(valid_segments, speaker_labels) if lbl == i
        ]
        if cluster_segments:
            pivot_segments_by_cluster[i].extend(
                random.sample(cluster_segments, k=min(3, len(cluster_segments)))
            )

    flat_pivots = [seg for segs in pivot_segments_by_cluster.values() for seg in segs]
    pivot_wavlm_embs = (
        extract_wavlm_embeddings_from_segments(
            waveform,
            encoder_small,
            flat_pivots,
            seg_len_sec=seg_len_small,
            n_samples=2,
            skip_short=False,
        )
        .squeeze(1)
        .numpy()
    )

    pivot_embs_by_cluster = {i: [] for i in range(n_speakers)}
    offset = 0
    for i in range(n_speakers):
        n = len(pivot_segments_by_cluster[i])
        pivot_embs_by_cluster[i].extend(pivot_wavlm_embs[offset : offset + n])
        offset += n

    # --- Compute WavLM pivot centers ---
    wavlm_pivot_centers = []
    for i in range(n_speakers):
        if len(pivot_embs_by_cluster[i]) > 0:
            wavlm_pivot_centers.append(np.mean(pivot_embs_by_cluster[i], axis=0))
        else:
            wavlm_pivot_centers.append(
                np.mean(pivot_wavlm_embs, axis=0)
            )  # global mean fallback
    wavlm_pivot_centers = np.stack(wavlm_pivot_centers)

    # --- Small-segment embeddings ---
    embeddings_small = (
        extract_wavlm_embeddings_from_segments(
            waveform,
            encoder_small,
            skipped_segments,
            seg_len_sec=seg_len_small,
            n_samples=1,
            skip_short=False,
        )
        .squeeze(1)
        .numpy()
    )

    if use_cosine_norm:
        n = np.linalg.norm(embeddings_small, axis=1, keepdims=True) + 1e-9
        embeddings_small_norm = embeddings_small / n
    else:
        embeddings_small_norm = embeddings_small

    # --- Cosine distance assignment ---
    pivot_norms = np.linalg.norm(wavlm_pivot_centers, axis=1, keepdims=True) + 1e-9
    emb_norms = np.linalg.norm(embeddings_small_norm, axis=1, keepdims=True) + 1e-9
    cosine_sim = (embeddings_small_norm @ wavlm_pivot_centers.T) / (
        emb_norms * pivot_norms.T
    )

    assigned_labels = np.argmax(cosine_sim, axis=1)
    assigned_conf = np.max(cosine_sim, axis=1)

    # --- Forced assignment (no missing) ---
    final_labels = assigned_labels  # every short segment gets a label

    labeled_small_segments = [
        {**seg, "speaker": f"Speaker {lbl}"}
        for seg, lbl in zip(skipped_segments, final_labels)
    ]

    merged_segments = sorted(
        labeled_large_segments + labeled_small_segments, key=lambda x: x["start"]
    )
    # print("Assigned Confident", cosine_sim)
    # print("pivot centers", wavlm_pivot_centers, pivot_norms)
    merged_labels = [seg["speaker"] for seg in merged_segments]

    # --- Optional visualization ---
    if plot:
        try:
            reducer = umap.UMAP(
                n_neighbors=10, min_dist=0.01, metric="cosine", random_state=42
            )
        except Exception:
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2)

        emb_large_2d = reducer.fit_transform(embeddings_large_norm)
        emb_small_2d = reducer.transform(embeddings_small_norm)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(
            emb_large_2d[:, 0], emb_large_2d[:, 1], c=speaker_labels, cmap="tab10", s=30
        )
        plt.title("Phase 1 — Large embeddings (coarse clusters)")

        plt.subplot(1, 2, 2)
        plt.scatter(
            emb_small_2d[:, 0],
            emb_small_2d[:, 1],
            c=assigned_labels,
            cmap="tab10",
            s=10,
            alpha=0.8,
        )
        plt.title("Phase 2 — Skipped short segments (fine clustering)")
        plt.tight_layout()
        plt.show()

    return {
        "segments_large": labeled_large_segments,
        "segments_small": labeled_small_segments,
        "merged_segments": merged_segments,
        "labels": merged_labels,
        "pivots": wavlm_pivot_centers,
    }

def der_benchmark(speech_segments, sample_rate, speaker_labels, meta):
    """
    Benchmark diarization performance with ground-truth metadata.

    Parameters
    ----------
    speech_segments : list[dict]
        Detected speech segments (with 'start'/'end' in samples).
    sample_rate : int
        Sampling rate of the audio (Hz).
    speaker_labels : list
        Predicted speaker labels per segment.
    meta : dict
        Ground-truth row containing keys:
        - "timestamps_start"
        - "timestamps_end"
        - "speakers"
    """
    required_keys = ["timestamps_start", "timestamps_end", "speakers"]

    # --- Validate metadata ---
    for key in required_keys:
        if key not in meta:
            print(f"[WARN] Missing key '{key}' in meta → skipping benchmark.")
            return None

    if not (meta["timestamps_start"] and meta["timestamps_end"]):
        print("[WARN] Empty timestamp lists in meta → skipping benchmark.")
        return None

    # --- Clean GT segments ---
    gt_segments = [
        (s, e) for s, e in zip(meta["timestamps_start"], meta["timestamps_end"])
    ]

    if len(gt_segments) == 0:
        print("[WARN] No valid GT segments after cleaning → skipping benchmark.")
        return None

    n_speaker_gt = len(set(meta.get("speakers", [])))

    # --- Convert detections ---
    if not speech_segments:
        print("[WARN] No detected speech segments provided.")
        return None

    detected_sec = [
        (s["start"] / sample_rate, s["end"] / sample_rate)
        for s in speech_segments
        if "start" in s and "end" in s
    ]

    if not detected_sec:
        print("[WARN] No valid detected segments (missing start/end).")
        return None

    # --- Evaluate ---
    recall, false_alarm = eval_detection(detected_sec, gt_segments)

    print(f"Recall={recall:.4f}, FA={false_alarm:.4f}")
    print(
        f"Found {len(set(speaker_labels))} speakers, " f"GT got {n_speaker_gt} speakers"
    )

    return recall, false_alarm


def map_speakers_by_overlap(gt_data, sys_data):
    gt_spks = sorted(set(spk for _, _, spk in gt_data))
    sys_spks = sorted(set(spk for _, _, spk in sys_data))

    # Build cost matrix: 1 - overlap ratio
    cost = np.ones((len(gt_spks), len(sys_spks)))

    for i, g in enumerate(gt_spks):
        for j, s in enumerate(sys_spks):
            overlap = 0.0
            for gs, ge, gsp in gt_data:
                if gsp != g:
                    continue
                for ss, se, ssp in sys_data:
                    if ssp != s:
                        continue
                    inter = max(0, min(ge, se) - max(gs, ss))
                    overlap += inter
            cost[i, j] = 1 - overlap

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {sys_spks[j]: gt_spks[i] for i, j in zip(row_ind, col_ind)}
    return mapping


def plot_diarization_timeline(
    speech_segments, sample_rate, speaker_labels, meta, max_time=None
):
    # Build GT and system data tuples: (start_time, end_time, speaker_id)
    gt_data = [
        (s, e, str(spk))
        for s, e, spk in zip(
            meta["timestamps_start"], meta["timestamps_end"], meta["speakers"]
        )
    ]
    sys_data = [
        (seg["start"] / sample_rate, seg["end"] / sample_rate, str(lbl))
        for seg, lbl in zip(speech_segments, speaker_labels)
    ]

    # Align system labels to ground truth speaker IDs if overlap mapping available
    mapping = map_speakers_by_overlap(gt_data, sys_data)
    sys_data = [(s, e, mapping.get(spk, spk)) for (s, e, spk) in sys_data]

    # Unified color palette for both GT and System
    all_spks = sorted(set(spk for _, _, spk in gt_data + sys_data))
    cmap = plt.cm.get_cmap("Dark2", len(all_spks))
    color_map = {spk: cmap(i) for i, spk in enumerate(all_spks)}

    # Start figure
    fig, ax = plt.subplots(figsize=(20, 5))
    y_gt, y_sys = 1, 0

    def draw(data, y):
        for s, e, k in data:
            if e <= s:
                continue  # skip broken or empty spans
            ax.add_patch(
                mpatches.Rectangle(
                    (s, y - 0.35),
                    e - s,
                    0.7,
                    color=color_map.get(k, "gray"),
                    alpha=1.0,
                    linewidth=0,
                )
            )

    # Draw both timelines
    draw(gt_data, y_gt)
    draw(sys_data, y_sys)

    # Axis setup
    ax.set_yticks([y_sys, y_gt])
    ax.set_yticklabels(["System", "Ground Truth"])
    max_t = max_time or max(e for e, _, _ in gt_data + sys_data)
    ax.set_xlim(0, max_t)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Unified legend
    legend = [mpatches.Patch(color=color_map[s], label=s) for s in all_spks]
    ax.legend(
        handles=legend,
        loc="upper center",
        ncol=min(len(all_spks), 10),
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )

    plt.tight_layout()
    plt.show()


def der_benchmark_full(
    speech_segments, sample_rate, speaker_labels, meta, frame_len=0.01
):
    """
    Compute DER with decomposition into Missed Speech, False Alarm, and Confusion.
    """
    noise_threshold = 0.2  # seconds
    # --- Ground Truth ---
    gt_segments_raw = [
        (s, e) for s, e in zip(meta["timestamps_start"], meta["timestamps_end"])
    ]
    gt_speakers_raw = meta["speakers"]

    # 🔸 Filter <100 ms segments (keep GT-speaker alignment)
    gt_segments, gt_speakers = (
        zip(
            *[
                (seg, spk)
                for seg, spk in zip(gt_segments_raw, gt_speakers_raw)
                if (seg[1] - seg[0]) >= noise_threshold
            ]
        )
        if len(gt_segments_raw) > 0
        else ([], [])
    )

    gt_frames, gt_spk_map = segments_to_frames(gt_segments, gt_speakers, frame_len)

    # --- System Output ---
    sys_segments_raw = [
        (s["start"] / sample_rate, s["end"] / sample_rate) for s in speech_segments
    ]
    sys_speakers_raw = speaker_labels

    # 🔸 Filter <100 ms segments (keep SYS-speaker alignment)
    sys_segments, sys_speakers = (
        zip(
            *[
                (seg, spk)
                for seg, spk in zip(sys_segments_raw, sys_speakers_raw)
                if (seg[1] - seg[0]) >= noise_threshold
            ]
        )
        if len(sys_segments_raw) > 0
        else ([], [])
    )

    sys_frames, sys_spk_map = segments_to_frames(sys_segments, sys_speakers, frame_len)
    # print(gt_segments[:5])
    # print(sys_segments[:5])
    # gt_dur = sum(e-s for s,e in gt_segments)
    # sys_dur = sum(e-s for s,e in sys_segments)
    # print(f"GT speech dur: {gt_dur:.1f}s, SYS speech dur: {sys_dur:.1f}s, ratio {sys_dur/gt_dur:.2f}")

    T = max(len(gt_frames), len(sys_frames))
    gt_frames = pad_or_trim(gt_frames, T)
    sys_frames = pad_or_trim(sys_frames, T)

    n_gt, n_sys = len(gt_spk_map), len(sys_spk_map)
    conf_mat = np.zeros((n_gt, n_sys))
    for t in range(T):
        g, s = gt_frames[t], sys_frames[t]
        if g is not None and s is not None:
            conf_mat[g, s] += 1
    mapping = {}
    if n_gt > 0 and n_sys > 0:
        cost = -conf_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {sys: gt for gt, sys in zip(row_ind, col_ind)}

    missed = fa = conf = 0
    total = sum(1 for g in gt_frames if g is not None)
    for t in range(T):
        g, s = gt_frames[t], sys_frames[t]
        if g is not None and s is None:
            missed += 1
        elif g is None and s is not None:
            fa += 1
        elif g is not None and s is not None:
            mapped_gt = mapping.get(s, None)
            if mapped_gt is None:
                missed += 1
                fa += 1
            elif mapped_gt != g:
                conf += 1
    der = (missed + fa + conf) / total if total > 0 else 0.0
    print(missed, fa, conf, total)
    print(
        f"DER={der:.4f} (Miss={missed/total:.4f}, FA={fa/total:.4f}, Conf={conf/total:.4f})"
    )
    plot_diarization_timeline(speech_segments, sample_rate, speaker_labels, meta)
    return der, missed / total, fa / total, conf / total


def label_diagnostics(
    speech_segments, sample_rate, speaker_labels, meta, frame_len=0.01
):
    """
    Diagnose diarization labelling issues.
    Returns: (diagnostics_dict, compact_summary)
    """
    gt_segments = list(zip(meta["timestamps_start"], meta["timestamps_end"]))
    gt_speakers = meta["speakers"]
    gt = expand_segments_to_frames(gt_segments, gt_speakers, frame_len)
    sys_segments = [
        (s["start"] / sample_rate, s["end"] / sample_rate) for s in speech_segments
    ]
    sys = expand_segments_to_frames(sys_segments, speaker_labels, frame_len)
    T = max(len(gt), len(sys))
    gt = np.resize(gt, T)
    sys = np.resize(sys, T)
    sys_ids = list(set([s for s in sys if s is not None]))
    gt_ids = list(set([g for g in gt if g is not None]))
    if not sys_ids or not gt_ids:
        return {
            "mapping": {},
            "quality": 0,
            "issues": [("EMPTY", "no labels")],
        }, "Empty input"
    cost = np.zeros((len(sys_ids), len(gt_ids)))
    for i, s in enumerate(sys_ids):
        for j, g in enumerate(gt_ids):
            cost[i, j] = -np.sum((sys == s) & (gt == g))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {sys_ids[i]: gt_ids[j] for i, j in zip(row_ind, col_ind)}
    sys_remap = np.array([mapping.get(s, None) for s in sys])
    mask = gt != None
    quality = np.mean(sys_remap[mask] == gt[mask])
    issues = []
    gt_runs = group_runs(gt)
    sys_runs = group_runs(sys_remap)
    if len(sys_runs) > len(gt_runs):
        issues.append(
            ("OSCILLATION", f"{len(sys_runs)} sys runs vs {len(gt_runs)} gt runs")
        )
    for s in sys_ids:
        covered = set(gt[sys == s])
        if len(covered) > 1:
            label = mapping.get(s, f"UNMAPPED_SYS_{s}")
            issues.append(("MERGE", f"sys {label} covers {covered}"))
    for g in gt_ids:
        sys_set = set(sys_remap[gt == g])
        sys_set.discard(None)
        if len(sys_set) > 1:
            label = g if g in gt_ids else f"UNMAPPED_GT_{g}"
            issues.append(("SPLIT", f"gt {label} split across {sys_set}"))
    frame_errors = extract_frame_errors(gt, sys_remap, frame_len=frame_len)
    diagnostics = {
        "mapping": mapping,
        "quality": float(quality),
        "issues": issues,
        "frame_errors": frame_errors,
    }
    summary = f"Quality={quality:.2f} | " + (
        " | ".join([f"{typ}:{msg}" for typ, msg in issues])
        if issues
        else "No major issues"
    )
    return diagnostics, summary


def convert_seconds_to_hhmmss(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def process_timestamps(segments, sample_rate):
    processed = []
    for seg in segments:
        start_hhmmss = convert_seconds_to_hhmmss(seg["start"] / sample_rate)
        end_hhmmss = convert_seconds_to_hhmmss(seg["end"] / sample_rate)
        processed.append(
            {
                "start": start_hhmmss,
                "end": end_hhmmss,
                "speaker": seg["speaker"],
                "text": seg.get("text", ""),
            }
        )
    return processed


def postprocess_diarization(client, diarized_segments):
    chunk_size = 10  # number of segments per chunk
    system_prompt = """
    You correct speaker-turn boundaries in diarization text.

    Rules:
    1. Only rearrange the existing tokens; Do not add, delete, or alter any token.
    2. Move words between adjacent segments if the sentence flow proves words clearly belong to the previous or next segment.
    3. Adjust the boundary between two turns when they’re obviously wrong.
    4. You must move tokens when a segment begins with a phrase that grammatically attaches to the previous segment.
    5. Preserve speaker IDs exactly.
   # 6. Always output valid JSON (double quotes for keys/values) no markdown, comments, or extra text.
    7. If rules conflict, prioritize structural correctness of the conversation while preserving all tokens.
    8. Input may include 1–2 previous segments for context. Use them for boundary correction but do not duplicate them in the final output.
    9. Preserve JSON structure exactly:
    [
        {"start": "...", "end": "...", "speaker": "...", "text": "..."},
        ...
    ]
    """

    user_prompt = """
    Input:
    [
    {"start": "00:00:38", "end": "00:00:40", "speaker": "Speaker 1", "text": "Okay. You know what"},
    {"start": "00:00:40", "end": "00:00:43", "speaker": "Speaker 1", "text": "I mean? I'll take notes this time. What? I'll take notes this"},
    {"start": "00:00:43", "end": "00:00:49", "speaker": "Speaker 1", "text": "time. I wish"},
    {"start": "00:00:49", "end": "00:00:53", "speaker": "Speaker 3", "text": "I could get this first thing and just sort of did a really sort of breathe. Sort of brief,"},
    {"start": "00:00:54", "end": "00:00:57", "speaker": "Speaker 1", "text": "you know, version of what we"},
    {"start": "00:00:57", "end": "00:00:58", "speaker": "Speaker 3", "text": "have to do."},
    ]
    """
    assistant_prompt = """
    [
    {"start": "00:00:38", "end": "00:00:41", "speaker": "Speaker 1", "text": "Okay. You know what I mean? I'll take notes this time. "},
    {"start": "00:00:41", "end": "00:00:44", "speaker": "Speaker 1", "text": "What? I'll take notes this time."},
    {"start": "00:00:44", "end": "00:00:53", "speaker": "Speaker 3", "text": "I wish I could get this first thing and just sort of did a really sort of breathe."},
    {"start": "00:00:53", "end": "00:00:58", "speaker": "Speaker 1", "text": "Sort of brief, you know, version of what we have to do."},
    ]
    """

    all_outputs = """"""

    for i in range(0, len(diarized_segments), chunk_size):
        chunk = diarized_segments[i : i + chunk_size]
        request_prompt = f"Input:\n{json.dumps(chunk, ensure_ascii=False, indent=2)}\n"
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prompt},
                    {
                        "role": "user",
                        "content": request_prompt,
                    },
                ],
                temperature=0.6,
            )
            processed_output = response.choices[0].message.content
            all_outputs += processed_output + "\n"
        except Exception as e:
            print(f"Error during post processing: {e}")
    print("Post-processing complete.")
    print(all_outputs)
    return all_outputs


def correction_asr(text: str) -> dict:
    """Send text to LM Studio (Qwen3-4B) and get entities as dict with strict JSON schema."""
    
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    system_prompt = """
    Bạn là mô-đun hiệu chỉnh văn bản ASR tiếng Việt, vận hành theo quy trình *ưu tiên ngữ âm*-*phonetic-first*.

    Nhiệm vụ:
    - Xác định và sửa các từ/cụm từ sai do ASR chuyển từ *âm* thành *chữ* (gần âm, sai dấu, sai chính tả, tách/ghép sai).
    - Chỉ sửa khi từ đó vô nghĩa, sai chính tả, hoặc không khớp ngữ cảnh cục bộ.
    - Bắt buộc chọn ứng viên (từ thay tế) dựa trên giống âm (nghiệm-nghiệp, hun-hôn, địp-điệp, việc-việt); chỉ dùng ngữ cảnh để chọn giữa các ứng viên gần âm.
    - Không đoán nghĩa, không viết lại câu, không thêm/bớt nội dung.

    Đầu vào:
    {"text"}

    Đầu ra (JSON):
    {
    "reasoning": "<mô tả ngắn quá trình sửa từ chọn âm đến ngữ cảnh>",
    "text": "<văn bản đã chỉnh>"
    }

    Hạn chế:
    - Không được diễn giải hay sáng tạo nội dung.
    - Không thay đổi mọi thứ ngoài phần sửa lỗi âm.
    - Không được tự ý nối, ghép các câu lại với nhau.
    """

    # 2. VÍ DỤ ONE-SHOT (Chain of Thought)
    # Chủ đề: Y tế (để khác biệt hoàn toàn với rác thải/xây dựng)
    # Lỗi giả định: "tiểu đường" -> "triều cường", "huyết áp" -> "huyết giáp", "insu-lin" -> "in su linh"
    user_prompt_example = """
    Thử một ví dụ:
    Hãy sửa lỗi đoạn văn bản sau:
    \"\"\"bệnh nhân ị biến chứng do bệnh triều cường tiếp hai chỉ số đường huyết tăng cao cần tiêm in su linh định kỳ để ổn định huyết giáp và tránh si thạ  mãn tính beget\"\"\"
    """

    assistant_prompt_example = """
    {
        "reasoning": "ị→bị: ['bị','vị'] → chọn 'bị' (gần âm + hợp ngữ cảnh);
        triều cường→tiểu đường: ['tiểu trường','tiểu đường'] → chọn 'tiểu đường';
        tiếp→tuýp: ['tiếp','tuýp'] → chọn 'tuýp' (gần âm + hợp ngữ cảnh);
        in su linh→insulin: ['in-su-lin','insulin'] → chọn 'insulin';
        huyết giáp→huyết áp: ['huyết áp','huyết giáp','biến áp'] → chọn 'huyết áp';
        si thạ→suy thận: ['suy thận','si thậ','si thạ'] → chọn 'suy thận';
        beget→beget: ['bi-ghét','bê-gét'] không đủ chắc chắn (có thể là tên/thuật ngữ/địa danh) nên giữ nguyên",
        "text": "Bệnh nhân bị biến chứng do bệnh tiểu đường tuýp 2 chỉ số đường huyết tăng cao cần tiêm insulin định kỳ để ổn định huyết áp và tránh suy thận mãn tính beget"
    }
    """

    # 3. INPUT THỰC TẾ
    user_prompt_actual = f"""
    Hoàn toàn không liên quan tới ví dụ trên.
    Hãy sửa lỗi đoạn văn bản sau:
    \"\"\"{text}\"\"\"
    """

    # 4. SCHEMA (Thêm trường reasoning)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "correction_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["reasoning", "text"],
            },
        },
    }

    payload = {
        "model": "qwen3-4b-thinking-2507",
        # "model": "qwen/qwen3-1.7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_example},
            {"role": "assistant", "content": assistant_prompt_example},
            {"role": "user", "content": user_prompt_actual},
        ],
        "response_format": schema,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()

        data = resp.json()
        raw_output = data["choices"][0]["message"]["content"]

        return json.loads(raw_output)

    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Error: {e}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by LLM", "raw": raw_output}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}", "raw": raw_output}

DATASET_DIR = "audio_out"

# ================================================================
# Device provisioning
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device)


# ================================================================
# VAD + Speaker Models
# ================================================================
vad_model = load_silero_vad(onnx=True)

speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="./pretrained_models/spkrec-ecapa",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)

wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
wavlm_model.eval()

def fast_ctc_beam_search(log_probs, beam_size=25, blank_id=0):
    """
    High-performance prefix beam search.
    log_probs: [T, V] numpy float32 of log-softmax logits.
    Returns: best decoded prefix as list of token ids.
    """
    T, V = log_probs.shape

    # Beam = { prefix(tuple): (pb, pnb) }
    beams = {(): (0.0, -np.inf)}

    for t in range(T):
        frame = log_probs[t]

        # Top-K pruning for speed
        topk_ids = np.argpartition(frame, -beam_size)[-beam_size:]
        topk_vals = frame[topk_ids]

        new_beams = {}

        for prefix, (pb, pnb) in beams.items():
            for token, lp in zip(topk_ids, topk_vals):

                    end = prefix[-1] if prefix else None

                    if token == end:
                        # same token (no prefix extension)
                        new_prefix = prefix
                        new_pnb = pnb + lp
                    else:
                        # append new token
                        new_prefix = prefix + (token,)
                        new_pnb = max(pb, pnb) + lp

                    nb_pb, nb_pnb = new_beams.get(new_prefix, (-np.inf, -np.inf))
                    new_beams[new_prefix] = (nb_pb, np.logaddexp(nb_pnb, new_pnb))

        # prune to beam_size
        beams = dict(sorted(
            new_beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True
        )[:beam_size])

    # final best
    best_prefix = max(
        beams.items(),
        key=lambda x: np.logaddexp(x[1][0], x[1][1])
    )[0]

    return best_prefix

# ================================================================
# VLSP2020 no-LM Loader
# ================================================================
def load_vlsp2020_model_no_lm(device="cuda"):
    """
    Loads Wav2Vec2-base-vi-vlsp2020 WITHOUT LM.
    Windows-compatible (no KenLM).
    """
    model_name = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020" # large version is too slow

    py_path = hf_hub_download(repo_id=model_name, filename="model_handling.py")
    module = SourceFileLoader("vlsp2020", py_path).load_module()

    model = module.Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    return processor, model


# ================================================================
# Unified Transcription Functions
# ================================================================
def transcribe_chunks_vlsp2020(waveform, sr, chunks, device="cuda"):
    """VLSP2020 — no LM."""
    processor, model = load_vlsp2020_model_no_lm(device)
    vocab = list(processor.tokenizer.get_vocab().keys())
    vocab = sorted(vocab, key=lambda x: processor.tokenizer.get_vocab()[x])
    # print(vocab)
    # blank_id = vocab.index('|')
    # banned_ids = [
    #     vocab.index("<pad>"),
    #     vocab.index("<unk>"),
    #     vocab.index("<s>"),
    #     vocab.index("</s>")
    # ]
    transcripts = []
    # return transcripts
    for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="VLSP2020 (no-LM)"):
        segment = waveform[:, c["start"] : c["end"]].cpu().numpy().squeeze()

        inputs = processor(
            segment, sampling_rate=sr, return_tensors="pt", padding="longest"
        )

        input_values = inputs.input_values.to(device)
        mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = model(input_values, attention_mask=mask).logits

        # without beamsearch and n-gram lm
        predicted_ids = torch.argmax(logits, dim=-1)

        text = processor.tokenizer.decode(
            predicted_ids[0].cpu().numpy(), skip_special_tokens=True
        ).strip()
        # print(text)
        # only with beamsearch

        # convert to log-probs
        # logits = logits[0].cpu()
        # log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # for bid in banned_ids:
        #     log_probs[:, bid] = -1e9
        # best_ids = fast_ctc_beam_search(log_probs.numpy(), beam_size=25, blank_id=blank_id)
        # print(len(best_ids))
        # text = processor.tokenizer.decode(
        #     best_ids, skip_special_tokens=True
        # ).strip()

        transcripts.append(
            {
                "chunk_id": i,
                "start": convert_seconds_to_hhmmss(c["start"] / sr) ,
                "end": convert_seconds_to_hhmmss(c["end"] / sr),
                "text": text,
                "words": None,
            }
        )

    return transcripts


# ================================================================
# +++ NEW +++ Wav2Vec2-Base-250H (no LM)
# ================================================================
def load_w2v2_250h(device="cuda"):
    model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def transcribe_chunks_w2v2_250h(waveform, sr, chunks, device="cuda"):
    """Wav2Vec2-base-250h — no LM."""
    processor, model = load_w2v2_250h(device)
    outputs = []

    for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="W2V2-250h"):
        segment = waveform[:, c["start"] : c["end"]].cpu().numpy().squeeze()

        inputs = processor(
            segment, sampling_rate=sr, return_tensors="pt", padding="longest"
        )

        iv = inputs.input_values.to(device)
        mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = model(iv, attention_mask=mask).logits

        ids = torch.argmax(logits, dim=-1)
        text = processor.decode(ids[0].cpu().numpy()).strip()

        outputs.append(
            {
                "chunk_id": i,
                "start": c["start"],
                "end": c["end"],
                "text": text,
                "words": None,
            }
        )

    return outputs


# ================================================================
# +++ NEW +++ PhoWhisper
# ================================================================
def load_phowhisper(device="cuda"):
    model_name = "vinai/PhoWhisper-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def transcribe_chunks_phowhisper(waveform, sr, chunks, device="cuda"):
    """PhoWhisper — seq2seq inference (no timestamps)."""
    processor, model = load_phowhisper(device)
    results = []

    for i, c in tqdm(enumerate(chunks), total=len(chunks), desc="PhoWhisper"):
        segment = waveform[:, c["start"] : c["end"]].cpu().numpy().squeeze()

        inputs = processor(
            segment, sampling_rate=sr, return_tensors="pt", padding="longest"
        ).to(device)

        with torch.no_grad():
            generated = model.generate(**inputs)

        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        results.append(
            {
                "chunk_id": i,
                "start": c["start"],
                "end": c["end"],
                "text": text,
                "words": None,
            }
        )

    return results


# ================================================================
# Fixed-window chunking
# ================================================================
def chunk_fixed_only(total_samples, sr, chunk_sec=60.0, overlap_sec=2.0):
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)

    chunks = []
    start = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)

        chunks.append({"start": start, "end": end, "segments": []})

        if end == total_samples:
            break

        start = end - overlap_samples

    return chunks

def segment_asr_text(
    text: str,
    window: int = 18,
    overlap: int = 6,
) -> list:
    """
    Segment raw ASR text (no punctuation) into stable micro-chunks
    for phonetic correction.

    - window: number of words per chunk
    - overlap: overlap size to preserve local context
    """

    # --- guard clauses ---
    if not text or text.strip() == "":
        return []

    # Tokenize by whitespace (ASR usually outputs clean tokens)
    words = text.strip().split()
    n = len(words)

    if n <= window:
        return [text.strip()]

    chunks = []
    i = 0

    while i < n:
        start = i
        end = min(i + window, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # advance with overlap
        i += (window - overlap)

        # safety: stop if no progress
        if i <= start:
            break

    return chunks

def merge_corrected_spans(spans: list, overlap: int = 6) -> str:
    """
    Merge overlap-corrected spans without duplicating words.
    Last span wins if corrections differ.

    spans: list of corrected text segments
    overlap: number of words overlapped in segmentation
    """
    if not spans:
        return ""

    merged_words = spans[0].split()

    for i in range(1, len(spans)):
        prev = merged_words
        curr = spans[i].split()

        # Take last 'overlap' words from prev
        tail_prev = prev[-overlap:] if len(prev) >= overlap else prev

        # Find where curr diverges from tail_prev
        k = 0
        while k < min(len(tail_prev), len(curr)) and tail_prev[k] == curr[k]:
            k += 1

        # Append the non-duplicate tail of current span
        merged_words.extend(curr[k:])

    return " ".join(merged_words)



client = Groq(api_key=dotenv.get_key(dotenv.find_dotenv(), "GROQ_API"))

if __name__ == "__main__":
    source_path = Path("./dợt-báo-cáo_1.wav")  # variable later

    if not source_path.exists():
        # Try MP4 fallback
        mp4_path = source_path.with_suffix(".mp4")
        if mp4_path.exists():
            print(f"[INFO] WAV not found. Converting MP4 → WAV: {mp4_path.name}")
            wav_path = mp4_to_wav(mp4_path)
            source_path = Path(wav_path)
        else:
            raise FileNotFoundError(
                f"Neither WAV nor MP4 found:\n" f" - {source_path}\n" f" - {mp4_path}"
            )

    waveform, sample_rate = load_audio(str(source_path))
    waveform = ensure_mono(waveform)
    num_samples = waveform.shape[-1]
    duration_sec = num_samples / sample_rate

    hours = int(duration_sec // 3600)
    minutes = int((duration_sec % 3600) // 60)
    seconds = int(duration_sec % 60)

    print(f"Audio duration: {hours:02d}:{minutes:02d}:{seconds:02d}")

    # speech_segments, labeled_segments, speaker_labels = diarization(waveform, sample_rate, vad_model, speaker_encoder, row)
    # diar_map = diarization(
    #     waveform,
    #     sample_rate,
    #     vad_model,
    #     speaker_encoder,
    #     wavlm_model,
    #     use_cosine_norm=True,
    #     plot=False,
    # )
    # speech_segments = diar_map["merged_segments"]
    # speaker_labels = diar_map["labels"]

    # chunks = chunk_by_silence_and_overlap(
    #     speech_segments, sample_rate, max_chunk=60, overlap=2
    # )
    chunks = chunk_fixed_only(waveform.shape[1], sample_rate)
    # print(chunks)

    # Choose model here:
    ### transcript = transcribe_chunks_ctc(waveform, sample_rate, chunks, model, processor)
    # transcript = transcribe_chunks_phowhisper(waveform, sample_rate, chunks)
    # transcript = transcribe_chunks_w2v2_250h(waveform, sample_rate, chunks)
    transcript = transcribe_chunks_vlsp2020(waveform, sample_rate, chunks)

    # print(transcript)

    # for chunk in tqdm(transcript):
    #     raw_text = chunk["text"].strip()
    #     if not raw_text:
    #         print("text is null")
    #         continue

    #     spans = segment_asr_text(raw_text, window=20, overlap=6)

    #     corrected_spans = []
    #     for span in spans:
    #         corrected = correction_asr(span)
    #         if corrected == "":
    #             print("Error in prompting")
    #         else:
    #             # print(corrected["reasoning"])
    #             pass
    #         corrected_spans.append(corrected["text"])

    #     # ← New smart merge
    #     final_text = merge_corrected_spans(corrected_spans, overlap=6)
    #     chunk["text"] = final_text
        
    #         # chunk["text"] = new_text["text"]
    #     # break
    #     # print(chunk)
    # # print(speech_segments)
    # # final_result = assign_speakers(transcript, speech_segments, sample_rate)
    # # diar_speech = process_timestamps(final_result, sample_rate)
    # # raw_speech_diarization: str = postprocess_diarization(client, diar_speech)
    save_transcriptions_json(
        transcript, Path("transcriptions") / f"trans_{source_path.stem}.json"
    )
