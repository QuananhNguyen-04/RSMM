# %% [markdown]
# # Information Retrieval and Summarization on Multiple Users Meeting
# 

# %%
import random
from datasets import load_dataset, Audio
from huggingface_hub import constants
import torchaudio
import torch
import torch.nn.functional as F
import os, json, io
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from tqdm import tqdm
import whisper
from whisper.model import Whisper
import warnings
warnings.filterwarnings("ignore", module="whisper.timing")
import umap
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import soundfile as sf
from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    VADIterator,
    collect_chunks,
)
from typing import Tuple, List, Dict
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference import EncoderClassifier
import numpy as np

# %%

DATASET_DIR = "audio_out"


# %%


# %% [markdown]
# ### Save file to local

# %%
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


# %%
def extract_embedding(waveform, sr, classifier: EncoderClassifier) -> np.ndarray:

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
    return embedding.squeeze(0).cpu().numpy()


def clean_segments(starts, ends, min_dur=0.2):
    segments = [(s, e) for s, e in zip(starts, ends) if e > s]
    segments.sort(key=lambda x: x[0])

    cleaned = []
    for seg in segments:
        if seg[1] - seg[0] < min_dur:
            continue  # skip too short
        if not cleaned:
            cleaned.append(seg)
        else:
            prev_start, prev_end = cleaned[-1]
            if seg[0] <= prev_end:  # overlap
                cleaned[-1] = (prev_start, max(prev_end, seg[1]))
            else:
                cleaned.append(seg)
    return cleaned


# %%
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
    max_time = max(
        max(e for _, e in ground_truth),
        max(e for _, e in detected)
    )

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

def is_titanet(model) -> bool:
    """
    Heuristic check if a model is Titanet.
    Works for NeMo Titanet (EncDecSpeakerLabelModel).
    """
    name = model.__class__.__name__.lower()
    mod = model.__class__.__module__.lower()
    return "titanet" in name or "nemo" in mod

def extract_embeddings_from_segments(
    waveform: torch.Tensor,
    classifier : EncDecSpeakerLabelModel | EncoderClassifier,
    speech_segments: list,
    seg_len_sec: float = 3.0,
    sample_rate: int = 16000,
    n_samples: int = 2,
    aggregate: bool = True,
) -> torch.Tensor:
    """
    Extract embeddings from diarization speech segments using Titanet (or ECAPA).

    Handles Titanet (NeMo) vs ECAPA (SpeechBrain) automatically.
    """
    seg_len = int(seg_len_sec * sample_rate)
    min_len = int(0.5 * sample_rate)  # Titanet requirement
    titan = is_titanet(classifier)

    all_embeddings = []
    for ts in speech_segments:
        start, end = ts["start"], ts["end"]
        segment = waveform[:, start:end]
        seg_len_actual = segment.shape[-1]

        # Pad short segments
        if seg_len_actual < (min_len if titan else seg_len):
            pad_len = (min_len if titan else seg_len) - seg_len_actual
            segment = torch.nn.functional.pad(segment, (0, pad_len))
            seg_len_actual = segment.shape[-1]

        candidates = []
        if seg_len_actual > seg_len:
            # center + random crops
            candidates.append(max(0, (seg_len_actual - seg_len) // 2))
            for _ in range(n_samples - 1):
                candidates.append(random.randint(0, seg_len_actual - seg_len))
        else:
            candidates.append(0)

        sub_embs = []
        for s in candidates:
            crop = segment[:, s:s+seg_len]
            # print(crop.shape)
            with torch.no_grad():
                if titan:
                    classifier.eval()
                    emb = classifier.forward(
                        input_signal=crop.to(classifier.device),
                        input_signal_length=torch.tensor([crop.shape[-1]]).to(classifier.device),
                    )
                    if isinstance(emb, tuple):
                        emb = emb[1]  # Titanet returns (logits, embeddings)
                else:
                    emb = classifier.encode_batch(crop.to(classifier.device))
            sub_embs.append(emb.cpu())

        # aggregate
        if aggregate:
            all_embeddings.append(torch.mean(torch.stack(sub_embs), dim=0))
        else:
            all_embeddings.extend(sub_embs)

    assert len(all_embeddings) > 0, "No embeddings were created"
    return torch.cat(all_embeddings, dim=0)

# def extract_embeddings_from_segments(
#     waveform, classifier, speech_timestamps,
#     seg_len_sec: float = 2.0, sample_rate: int = 16000,
#     n_samples: int = 2, aggregate: bool = True
# ) -> torch.Tensor:
#     """
#     Extract embeddings with optional sub-sampling inside long segments.

#     Args:
#         waveform: Tensor [1, num_samples]
#         classifier: SpeechBrain encoder (e.g. ECAPA)
#         speech_timestamps: list of dicts with {"start", "end"} in samples
#         seg_len_sec: length of window in seconds
#         sample_rate: waveform sample rate
#         n_samples: how many sub-windows to take per long segment
#         aggregate: if True, average embeddings across sub-windows
#                    else return all embeddings separately
#     """
#     seg_len = int(seg_len_sec * sample_rate)
#     all_embeddings = []

#     for ts in speech_timestamps:
#         start, end = ts["start"], ts["end"]
#         segment = waveform[:, start:end]
#         seg_len_actual = segment.shape[-1]

#         if seg_len_actual < seg_len:
#             # Pad short segments
#             pad_len = seg_len - seg_len_actual
#             segment = torch.nn.functional.pad(segment, (0, pad_len))

#             with torch.no_grad():
#                 emb = classifier.encode_batch(segment.to(classifier.device))
#             all_embeddings.append(emb.cpu())
#             continue

#         # ---- Pick n_samples windows from the segment ----
#         candidates = []
#         # Always take the center
#         center_start = max(0, (seg_len_actual - seg_len) // 2)
#         candidates.append(center_start)

#         # Optionally take one more at random
#         for _ in range(n_samples - 1):
#             rand_start = random.randint(0, seg_len_actual - seg_len)
#             candidates.append(rand_start)

#         sub_embs = []
#         for s in candidates:
#             crop = segment[:, s:s+seg_len]
#             with torch.no_grad():
#                 emb = classifier.encode_batch(crop.to(classifier.device))
#             sub_embs.append(emb.cpu())

#         # Aggregate or keep separately
#         if aggregate:
#             emb_final = torch.mean(torch.stack(sub_embs), dim=0)
#             all_embeddings.append(emb_final)
#         else:
#             all_embeddings.extend(sub_embs)

#     assert len(all_embeddings) > 0, "No embeddings were created"
#     return torch.cat(all_embeddings, dim=0)


# %%
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


# %%
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


# %%
def group_segments(segments, labels, max_gap=0.8, max_len=30.0, sr=16000):
    """
    Merge speech segments from the same speaker if they are close enough.
    - max_gap: maximum silence (in seconds) to allow merging
    - max_len: maximum total segment length (in seconds)
    """
    grouped = []
    cur_start, cur_end = None, None
    cur_label = None

    for seg, label in zip(segments, labels):
        start, end = seg["start"] / sr, seg["end"] / sr

        if cur_label is None:
            # start new group
            cur_start, cur_end, cur_label = start, end, label
            continue

        if (
            label == cur_label
            and (start - cur_end) <= max_gap
            and (end - cur_start) <= max_len
        ):
            # extend current group
            cur_end = end
        else:
            grouped.append({"speaker": cur_label, "start": cur_start, "end": cur_end})
            cur_start, cur_end, cur_label = start, end, label

    if cur_label is not None:
        grouped.append({"speaker": cur_label, "start": cur_start, "end": cur_end})

    return grouped


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
        if segment.max() > 1.0:
            segment = segment / max(1e-9, abs(segment).max())

        # Run Whisper with timestamps
        result = asr_model.transcribe(
            segment,
            fp16=False,  # safer on CPU/small GPU
            word_timestamps=True,  # return per-word timestamps
            beam_size=7,  # beam search for stability
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
    max_chunk=30.0,
    overlap=5.0,
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


# %%
def diarization(waveform, sample_rate, vad_model, encoder, meta=None):
    # --- Voice Activity Detection (VAD) ---
    # speech_segments = get_speech_timestamps(
    #     waveform,
    #     vad_model,
    #     sampling_rate=sample_rate,
    #     threshold=0.1,
    #     # neg_threshold=0.1,
    #     min_speech_duration_ms=100,
    #     min_silence_duration_ms=50,
    # )
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
    # speech_segments = get_speech_timestamps(
    #     waveform,
    #     vad_model,
    #     sampling_rate=sample_rate,
    #     threshold=0.2,
    #     # neg_threshold=0.1,
    #     min_silence_duration_ms=100,
    # )

    # --- Speaker Embeddings ---
    embeddings = extract_embeddings_from_segments(
        waveform, encoder, speech_segments, n_samples=1, seg_len_sec=2
    )  # tensor [num_segments, 1, emb_dim]
    # embeddings = extract_embeddings_titanet(waveform, encoder, speech_segments)
    # Flatten tensor for clustering
    # print(embeddings.shape)
    embeddings = embeddings.squeeze(1).numpy()
    # --- Clustering (Speaker Diarization) ---
    speaker_labels, thr, score = agglomerative_k_search(
        embeddings,
        linkage_method="average",
        metric="cosine",
        min_clusters=2,
        max_clusters=20,
    )

    # speaker_labels, cluster_ids, emb_2d, label2id, res = umap_cluster_embeddings(
    #     embeddings, reducer_dim=10, gt_labels=meta.get("speakers", [])
    # )
    # label_ids = res["numeric_labels"]
    # print("ARI:", res["ari"], " | NMI:", res["nmi"])
    # speaker_labels, k, eigvals, gaps = spectral_cluster_eigengap(embeddings)
    # print("Best eigen gap score:", gaps)
    # print("Labels:", speaker_labels[:20])

    # Reduce to 2D for visualization
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    # emb_2d = reducer.fit_transform(embeddings)

    # plt.figure(figsize=(8,6))
    # plt.scatter(emb_2d[:,0], emb_2d[:,1], c=label_ids, cmap="tab10", s=10)
    # plt.title("UMAP projection of embeddings")
    # plt.show()
    # --- Attach labels to speech segments ---
    labeled_segments = [
        {
            "speaker": f"Speaker {label}",
            "start": seg["start"],  # in samples
            "end": seg["end"],  # in samples
        }
        for seg, label in zip(speech_segments, speaker_labels)
    ]

    return speech_segments, labeled_segments, speaker_labels

# %%
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
    gt_segments = [(s,e) for s, e in zip(
        meta["timestamps_start"], meta["timestamps_end"])]

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
        f"Found {len(set(speaker_labels))} speakers, "
        f"GT got {n_speaker_gt} speakers"
    )

    return recall, false_alarm


# %%
def segments_to_frames(segments, speakers, frame_len=0.01):
    """
    Convert segment-level annotations into frame-level speaker arrays.

    Parameters
    ----------
    segments : list of (start, end)
        Segment boundaries in seconds.
    speakers : list of str
        Speaker labels, aligned with segments.
    frame_len : float
        Frame size in seconds (default=0.01 → 10 ms).

    Returns
    -------
    frames : np.ndarray of shape (T,)
        Each element is an integer speaker ID, or None for silence.
    spk_to_id : dict
        Mapping from speaker label to integer ID.
    """
    assert len(segments) == len(speakers), "Segments and speakers must align"
    print(segments[0], speakers[0])
    # Map speakers to IDs
    unique_speakers = list(dict.fromkeys(speakers))  # preserve order
    spk_to_id = {spk: i for i, spk in enumerate(unique_speakers)}

    # Determine max time
    max_time = max(end for _, end in segments)
    T = int(np.ceil(max_time / frame_len))

    frames = np.full(T, None)  # default = silence

    # Fill frames with speaker IDs
    for (start, end), spk in zip(segments, speakers):
        sid = spk_to_id[spk]
        start_idx = int(np.floor(start / frame_len))
        end_idx = int(np.ceil(end / frame_len))
        frames[start_idx:end_idx] = sid

    print(frames)
    return frames, spk_to_id

def pad_or_trim(arr, target_len, fill=None):
    """
    Pad or trim a frame array to a target length.

    Parameters
    ----------
    arr : np.ndarray
        Input array (1D).
    target_len : int
        Desired output length.
    fill : any
        Fill value if padding is needed (default=None).

    Returns
    -------
    out : np.ndarray
        Array of length `target_len`.
    """
    cur_len = len(arr)

    if cur_len == target_len:
        return arr

    if cur_len < target_len:
        pad_len = target_len - cur_len
        pad_vals = np.full(pad_len, fill, dtype=object)
        return np.concatenate([arr, pad_vals])

    return arr[:target_len]

# %%
def der_benchmark_full(speech_segments, sample_rate, speaker_labels, meta, frame_len=0.01):
    """
    Compute DER with decomposition into Missed Speech, False Alarm, and Confusion.
    Robust to unmapped speakers and arbitrary speaker labels.
    """

    # --- Ground-truth prep ---
    gt_segments = [(s, e) for s, e in zip(meta["timestamps_start"], meta["timestamps_end"])]
    gt_speakers = meta["speakers"]

    # Map GT speakers to contiguous IDs
    gt_spk_map = {spk: i for i, spk in enumerate(sorted(set(gt_speakers)))}
    gt_frames, gt_spk_map = segments_to_frames(gt_segments, gt_speakers, frame_len)

    # --- System prep ---
    sys_segments = [(s["start"] / sample_rate, s["end"] / sample_rate) for s in speech_segments]

    sys_spk_map = {spk: i for i, spk in enumerate(sorted(set(speaker_labels)))}
    sys_frames, sys_spk_map = segments_to_frames(sys_segments, speaker_labels, frame_len)
    # --- Align frame length ---
    T = max(len(gt_frames), len(sys_frames))
    gt_frames = pad_or_trim(gt_frames, T)
    sys_frames = pad_or_trim(sys_frames, T)

    n_gt = len(gt_spk_map)
    n_sys = len(sys_spk_map)
    conf_mat = np.zeros((n_gt, n_sys))

    # --- Build confusion matrix ---
    for t in range(T):
        g, s = gt_frames[t], sys_frames[t]
        if g is not None and s is not None:
            conf_mat[g, s] += 1

    # --- Optimal mapping (Hungarian) ---
    if n_gt > 0 and n_sys > 0:
        cost = -conf_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {sys: gt for gt, sys in zip(row_ind, col_ind)}
    else:
        mapping = {}

    # --- Frame-level scoring ---
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
            if mapped_gt is None:
                # system speaker not mapped to any GT → count as miss+FA
                missed += 1
                fa += 1
            elif mapped_gt != g:
                conf += 1

    der = (missed + fa + conf) / total if total > 0 else 0.0
    print(missed, fa, conf, total)
    print(
        f"DER={der:.4f} (Miss={missed/total:.4f}, FA={fa/total:.4f}, Conf={conf/total:.4f})"
    )
    return der, missed / total, fa / total, conf / total

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
    """
    Extract mismatched segments (GT vs SYS) with local neighborhood.
    Compacts into time intervals instead of frame ranges.
    
    Args:
        gt, sys_remap : arrays of labels (frame-level).
        frame_len     : seconds per frame.
        window        : frames of neighborhood to include.
        merge_gap     : max seconds gap to merge consecutive mismatches with same (gt, sys).
    """
    T = len(gt)
    raw = []

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



def label_diagnostics(speech_segments, sample_rate, speaker_labels, meta, frame_len=0.01):
    """
    Diagnose diarization labelling issues.
    Returns: (diagnostics_dict, compact_summary)
    """

    # --- GT frames ---
    gt_segments = list(zip(meta["timestamps_start"], meta["timestamps_end"]))
    gt_speakers = meta["speakers"]
    gt = expand_segments_to_frames(gt_segments, gt_speakers, frame_len)

    # --- SYS frames ---
    sys_segments = [(s["start"]/sample_rate, s["end"]/sample_rate) for s in speech_segments]
    sys = expand_segments_to_frames(sys_segments, speaker_labels, frame_len)

    # --- Align lengths ---
    T = max(len(gt), len(sys))
    gt = np.resize(gt, T)
    sys = np.resize(sys, T)

    # --- Hungarian mapping ---
    sys_ids = list(set([s for s in sys if s is not None]))
    gt_ids = list(set([g for g in gt if g is not None]))
    if not sys_ids or not gt_ids:
        return {"mapping": {}, "quality": 0, "issues": [("EMPTY","no labels")]}, "Empty input"

    cost = np.zeros((len(sys_ids), len(gt_ids)))
    for i, s in enumerate(sys_ids):
        for j, g in enumerate(gt_ids):
            cost[i, j] = -np.sum((sys == s) & (gt == g))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {sys_ids[i]: gt_ids[j] for i, j in zip(row_ind, col_ind)}

    # Remap sys → gt space
    sys_remap = np.array([mapping.get(s, None) for s in sys])

    # --- Quality ---
    mask = gt != None
    quality = np.mean(sys_remap[mask] == gt[mask])

    # --- Issues ---
    issues = []

    # Oscillation
    gt_runs = group_runs(gt)
    sys_runs = group_runs(sys_remap)
    if len(sys_runs) > len(gt_runs):
        issues.append(("OSCILLATION", f"{len(sys_runs)} sys runs vs {len(gt_runs)} gt runs"))

    # Merge check
    for s in sys_ids:
        covered = set(gt[sys == s])
        if len(covered) > 1:
            label = mapping.get(s, f"UNMAPPED_SYS_{s}")
            issues.append(("MERGE", f"sys {label} covers {covered}"))

    # Split check
    for g in gt_ids:
        sys_set = set(sys_remap[gt == g])
        sys_set.discard(None)
        if len(sys_set) > 1:
            label = g if g in gt_ids else f"UNMAPPED_GT_{g}"
            issues.append(("SPLIT", f"gt {label} split across {sys_set}"))

    # --- Frame-level mismatches ---
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

# %% [markdown]
# ## Load model for Speech2Text

# %%
device="cpu"
device="cuda"

# %%
# asr_model: Whisper = whisper.load_model(
#     "small", device=device
# )  # or "medium", "large-v2"

# %%
vad_model = load_silero_vad(onnx=True)
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="./pretrained_models/spkrec-ecapa",
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)

# speaker_encoder = EncDecSpeakerLabelModel.from_pretrained("titanet_large")

# %% [markdown]
# ## Load dataset

# %%
ds = load_dataset("diarizers-community/ami", "sdm", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

# # %%
# print(ds.features)  # schema of all features
# # Inspect a single row
# row = ds[0]
# print("Keys:", row.keys())
# print("Audio field keys:", row["audio"].keys())
# print("Audio path:", row["audio"]["path"])
# print("Audio bytes length:", len(row["audio"]["bytes"]))

# %% [markdown]
# ## Process Dataset

# %%
avg_miss = avg_fa = avg_conf = 0

for idx, row in enumerate(ds):
    print(idx)
    # audio_path = row["audio"]["path"]   # local path in HF cache
    # full_path = os.path.join(base_dir, audio_path)
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
    
    speech_segments, labeled_segments, speaker_labels = diarization(waveform, sample_rate, vad_model, speaker_encoder, row)
    der_benchmark(speech_segments, sample_rate, speaker_labels, row)
    der, miss_rate, fa_rate, conf_rate = der_benchmark_full(speech_segments, sample_rate, speaker_labels, row)
    diag, summary = label_diagnostics(speech_segments, sample_rate, speaker_labels, row)
    # print(diag)
    print(summary)
    avg_miss += miss_rate
    avg_fa += fa_rate
    avg_conf += conf_rate

    # chunks = chunk_by_silence_and_overlap(speech_segments, sample_rate, max_chunk=30, overlap=3)
    # print(chunks)
    # transcript = transcribe_chunks(waveform, sample_rate, chunks, asr_model)
    # final_result = assign_speakers(transcript, labeled_segments, sample_rate)
    # break  # demo with first sample
    # if idx > 5:
    # break
    print(avg_miss / (idx+1), avg_fa /(idx+1), avg_conf/(idx+1))
data_size = idx + 1
print(avg_miss / data_size, avg_fa /data_size, avg_conf/data_size)

# # %%
# speaker_labels

# # %%
# diag.keys()

# # %%
# diag["frame_errors"]

# %%



