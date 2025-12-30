import io, json, random, warnings
from typing import Tuple, List, Dict

# Core Scientific / ML
import numpy as np
import torch
import torchaudio
import dotenv

# Hugging Face / Datasets
from datasets import load_dataset, Audio
from huggingface_hub import constants
from transformers.models.wavlm import WavLMModel

# Speech / Audio Models
import whisper
from whisper.model import Whisper
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference import EncoderClassifier
from groq import Groq

# Audio Processing / VAD
from silero_vad import (
    load_silero_vad,
    # read_audio,
    get_speech_timestamps,
    # save_audio,
    # VADIterator,
    # collect_chunks,
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

# Visualization / Progress
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local Project Utilities
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

import matplotlib.patches as mpatches




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
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    L = np.eye(len(sim)) - D_inv_sqrt @ sim @ D_inv_sqrt

    # Step 4: eigen decomposition
    eigvals, _ = eigh(L)
    eigvals = np.sort(eigvals)[:max_clusters+1]

    # Step 5: eigen-gap
    gaps = np.diff(eigvals)
    search_region = gaps[min_clusters-1:max_clusters-1]
    best_idx = np.argmax(search_region)
    k_opt = best_idx + min_clusters

    # Step 6: spectral clustering with k_opt
    clustering = SpectralClustering(
        n_clusters=k_opt,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=0
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

    print(len(set(gt_labels)))
    # 2) cluster count
    if n_clusters is None:
        if gt_labels is not None:
            n_clusters = len(set(gt_labels))
        else:
            raise ValueError("Need n_clusters or gt_labels to infer cluster count")

    # 3) clustering
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_ids = km.fit_predict(emb_reduced)   # numeric 0..K-1 (but order not guaranteed)

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
        segment = waveform[:, c["start"]:c["end"]].cpu().numpy()
        segment = segment.squeeze().astype("float32")

        # Normalize if needed
        # if segment.max() > 1.0:
        #     segment = segment / max(1e-9, abs(segment).max())

        # Run Whisper with timestamps
        result = asr_model.transcribe(
            segment,
            fp16=True,  # safer on CPU/small GPU
            word_timestamps=True,  # return per-word timestamps
            beam_size=10,  # beam search for stability
            # temperature=(0, 0.1, 0.2, 0.4, 0.8),  # deterministic output
            compression_ratio_threshold=1.8,
            language='en'
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
                    (s, e)
                    for (s, e) in c["segments"]
                    if e > start and s < end
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
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42)
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)

    emb_large_2d = reducer.fit_transform(emb_large)
    emb_small_2d = reducer.transform(emb_small)
    print("set size and items", len(set(labels_large)), len(set(labels_small)), set(labels_large), set(labels_small))
    norm_labels_large = normalize_labels(labels_large)
    norm_labels_small = normalize_labels(labels_small)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(emb_large_2d[:, 0], emb_large_2d[:, 1], c=norm_labels_large, cmap="tab10", s=30)
    plt.title("Phase 1 — Large embeddings (coarse clusters)")

    plt.subplot(1, 2, 2)
    plt.scatter(emb_small_2d[:, 0], emb_small_2d[:, 1], c=norm_labels_small, cmap="tab10", s=10, alpha=0.8)
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

    seg_len_large = 1.5
    seg_len_small = 0.7

    # 🔸 capture which segments were skipped due to length
    valid_segments = []
    skipped_segments = []

    # seg_len = int(seg_len_large * sample_rate)
    min_len = int(seg_len_small* sample_rate)  # same as min_len_ratio * sample_rate

    # partition segments before extraction
    for ts in speech_segments:
        start, end = ts["start"], ts["end"]
        seg_len_actual = end - start
        if seg_len_actual < min_len:
            skipped_segments.append(ts)
        else:
            valid_segments.append(ts)

    embeddings_large = (
        extract_embeddings_from_segments(
            waveform,
            encoder_large,
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
    speaker_labels, _, _, _ = spectral_cluster_eigengap(embeddings_large_norm, min_clusters=3, max_clusters=10)
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
            n_samples=1,
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

def map_speakers_by_overlap(gt_data, sys_data):
    gt_spks = sorted(set(spk for _,_,spk in gt_data))
    sys_spks = sorted(set(spk for _,_,spk in sys_data))

    # Build cost matrix: 1 - overlap ratio
    cost = np.ones((len(gt_spks), len(sys_spks)))

    for i, g in enumerate(gt_spks):
        for j, s in enumerate(sys_spks):
            overlap = 0.0
            for gs, ge, gsp in gt_data:
                if gsp != g: continue
                for ss, se, ssp in sys_data:
                    if ssp != s: continue
                    inter = max(0, min(ge, se) - max(gs, ss))
                    overlap += inter
            cost[i, j] = 1 - overlap

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {sys_spks[j]: gt_spks[i] for i, j in zip(row_ind, col_ind)}
    return mapping


def plot_diarization_timeline(speech_segments, sample_rate, speaker_labels, meta, max_time=None):
    # Build GT and system data tuples: (start_time, end_time, speaker_id)
    gt_data = [
        (s, e, str(spk))
        for s, e, spk in zip(meta["timestamps_start"], meta["timestamps_end"], meta["speakers"])
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
    gt_segments_raw = [(s, e) for s, e in zip(meta["timestamps_start"], meta["timestamps_end"])]
    gt_speakers_raw = meta["speakers"]

    # 🔸 Filter <100 ms segments (keep GT-speaker alignment)
    gt_segments, gt_speakers = zip(*[
        (seg, spk)
        for seg, spk in zip(gt_segments_raw, gt_speakers_raw)
        if (seg[1] - seg[0]) >= noise_threshold
    ]) if len(gt_segments_raw) > 0 else ([], [])

    gt_frames, gt_spk_map = segments_to_frames(gt_segments, gt_speakers, frame_len)

    # --- System Output ---
    sys_segments_raw = [(s["start"] / sample_rate, s["end"] / sample_rate) for s in speech_segments]
    sys_speakers_raw = speaker_labels

    # 🔸 Filter <100 ms segments (keep SYS-speaker alignment)
    sys_segments, sys_speakers = zip(*[
        (seg, spk)
        for seg, spk in zip(sys_segments_raw, sys_speakers_raw)
        if (seg[1] - seg[0]) >= noise_threshold
    ]) if len(sys_segments_raw) > 0 else ([], [])

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
        processed.append({
            "start": start_hhmmss,
            "end": end_hhmmss,
            "speaker": seg["speaker"],
            "text": seg.get("text", "")
        })
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
    6. Always output valid JSON (double quotes for keys/values) no markdown, comments, or extra text.
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

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

asr_model: Whisper = whisper.load_model(
    "small", device=device
)

vad_model = load_silero_vad(onnx=True)
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="./pretrained_models/spkrec-ecapa",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)
torch_device = torch.device(device)

wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
wavlm_model.eval()
# speaker_encoder = EncDecSpeakerLabelModel.from_pretrained("titanet_large")


ds = load_dataset("diarizers-community/ami", "sdm", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
client = Groq(api_key=dotenv.get_key(dotenv.find_dotenv(), "GROQ_API"))

avg_miss = avg_fa = avg_conf = 0
idx = 0
for idx, row in enumerate(ds):
    # audio_path = row["audio"]["path"]   # local path in HF cache
    # full_path = os.path.join(base_dir, audio_path)
    # if idx in [0, 1]:
    #     continue
    audio_bytes = row["audio"]["bytes"]
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    # resample if needed
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    waveform = ensure_mono(waveform)
    num_samples = waveform.shape[-1]
    duration_sec = num_samples / sample_rate

    hours = int(duration_sec // 3600)
    minutes = int((duration_sec % 3600) // 60)
    seconds = int(duration_sec % 60)

    print(f"Audio duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # speech_segments, labeled_segments, speaker_labels = diarization(waveform, sample_rate, vad_model, speaker_encoder, row)
    diar_map = diarization(waveform, sample_rate, vad_model, speaker_encoder, wavlm_model, row, use_cosine_norm=False, plot=False)
    speech_segments = diar_map["merged_segments"]
    speaker_labels = diar_map["labels"]
    # der_benchmark(speech_segments, sample_rate, speaker_labels, row)
    der, miss_rate, fa_rate, conf_rate = der_benchmark_full(speech_segments, sample_rate, speaker_labels, row)
    # diag, summary = label_diagnostics(speech_segments, sample_rate, speaker_labels, row)
    # print(diag)
    # print(summary)
    avg_miss += miss_rate
    avg_fa += fa_rate
    avg_conf += conf_rate

    chunks = chunk_by_silence_and_overlap(speech_segments, sample_rate, max_chunk=60, overlap=2)
    # print(chunks)
    transcript = transcribe_chunks(waveform, sample_rate, chunks, asr_model)
    final_result = assign_speakers(transcript, speech_segments, sample_rate)
    diar_speech = process_timestamps(final_result, sample_rate)
    raw_speech_diarization: str = postprocess_diarization(client, diar_speech)
    final_speech_diarization = concat_json_arrays(raw_speech_diarization)
    save_transcriptions_json(final_speech_diarization, f"transcriptions/final_transcriptions{idx}.json")
    # break  # demo with first sample
    if idx > 1:
        break
data_size = idx + 1
print(avg_miss / data_size, avg_fa /data_size, avg_conf/data_size)