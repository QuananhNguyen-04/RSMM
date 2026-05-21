import os
import re
import json
import math
from collections import defaultdict

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


# =========================================================
# CONFIG
# =========================================================

SIM_THRESHOLD = 0.50

MODEL_NAME = "all-MiniLM-L6-v2"

# weighted importance
INFO_WEIGHTS = {
    "participant": 1.5,
    "topic": 1.0,
    "meeting": 2.0
}


# =========================================================
# LOAD MODEL
# =========================================================

embedder = SentenceTransformer(MODEL_NAME)

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


# =========================================================
# HELPERS
# =========================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text.strip()


def tokenize(text):
    return normalize(text).split()


def token_count(text):
    return len(tokenize(text))


def flatten_gt_participant(gt_participant):
    """
    GT:
    {
        "A": [...],
        "B": [...]
    }
    """
    units = []

    for speaker, items in gt_participant.items():
        for t in items:
            if t.strip():
                units.append({
                    "type": "participant",
                    "speaker": speaker,
                    "text": t
                })

    return units


def flatten_gt_topics(gt_topics):
    units = []

    for t in gt_topics:
        desc = t.get("description", "")
        text = t.get("text", "")

        if text.strip():
            units.append({
                "type": "topic",
                "description": desc,
                "text": text
            })

    return units


def flatten_gt_meeting(gt_meeting):
    units = []

    for t in gt_meeting:
        if t.strip():
            units.append({
                "type": "meeting",
                "text": t
            })

    return units


# =========================================================
# LOAD PREDICTIONS
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_prediction_texts(
    speaker_json,
    topic_json,
    meeting_json
):
    speaker_text = []
    topic_text = []
    meeting_text = []

    # -------------------------
    # speaker summaries
    # -------------------------

    for item in speaker_json:
        s = item.get("summary", "").strip()

        if s:
            speaker_text.append(s)

    # -------------------------
    # topic summaries
    # -------------------------

    for item in topic_json:
        s = item.get("summary", "").strip()

        if s:
            topic_text.append(s)

    # -------------------------
    # meeting summary
    # -------------------------

    meeting_sum = meeting_json.get("summary", "").strip()

    if meeting_sum:
        meeting_text.append(meeting_sum)

    return {
        "participant": speaker_text,
        "topic": topic_text,
        "meeting": meeting_text
    }


# =========================================================
# ICS
# =========================================================

def split_sentences(text):

    sents = re.split(r"[.!?]+", text)

    return [
        normalize(s)
        for s in sents
        if len(normalize(s)) > 5
    ]


def flatten_pred_chunks(pred_texts):
    """
    Split predicted summaries into smaller chunks.
    """

    chunks = []

    for t in pred_texts:
        chunks.extend(split_sentences(t))

    return chunks

def is_low_information(text):

    text = normalize(text)

    if len(text.split()) < 4:
        return True
    
    return False


def compute_overlap(a, b):

    a_words = set(normalize(a).split())
    b_words = set(normalize(b).split())

    if len(a_words) == 0:
        return 0.0

    return len(a_words & b_words) / len(a_words)


def compute_layer_ics(
    gt_units,
    pred_texts,
    layer_name="unknown"
):

    pred_chunks = flatten_pred_chunks(pred_texts)

    if len(pred_chunks) == 0:
        return {
            "layer": layer_name,
            "score": 0.0,
            "details": []
        }

    pred_embs = embedder.encode(
        pred_chunks,
        convert_to_tensor=True
    )

    total_score = 0.0
    total_weight = 0.0

    details = []

    for u in gt_units:

        raw_text = u["text"]
        gt_chunks = split_sentences(raw_text)
        gt_chunks = [x for x in gt_chunks if not is_low_information(x)]
        if len(gt_chunks) == 0:
            continue
        weight = INFO_WEIGHTS.get(
            u["type"],
            1.0
        )

        for gt_chunk in gt_chunks:

            gt_emb = embedder.encode(
                gt_chunk,
                convert_to_tensor=True
            )

            overlaps = []

            # ---------------------------------
            # compute lexical overlaps first
            # ---------------------------------

            for pred_chunk in pred_chunks:

                overlap = compute_overlap(
                    gt_chunk,
                    pred_chunk
                )

                overlaps.append(overlap)

            overlaps = np.array(overlaps)

            # ---------------------------------
            # retrieve top-k by overlap
            # ---------------------------------

            k = min(3, len(pred_chunks))

            top_idxs = overlaps.argsort()[::-1][:k]

            top_scores = []

            top_pred_chunks = []

            # ---------------------------------
            # refine using semantic similarity
            # ---------------------------------

            for idx in top_idxs:

                pred_chunk = pred_chunks[idx]

                pred_emb = pred_embs[idx]

                sim_score = util.cos_sim(
                    gt_emb,
                    pred_emb
                ).item()

                overlap = overlaps[idx]

                # -----------------------------
                # overlap-first scoring
                # -----------------------------

                combined = (
                    0.7 * overlap
                    + 0.3 * sim_score
                )

                top_scores.append(combined)

                top_pred_chunks.append(pred_chunk)

            # ---------------------------------
            # weighted top-k coverage
            # ---------------------------------

            weights = [0.5, 0.3, 0.2]

            coverage = 0.0

            for i in range(len(top_scores)):
                coverage += weights[i] * top_scores[i]

            coverage = min(coverage, 1.0)

            # max_idx = sims.argmax().item()
            # best_sim = sims[max_idx].item()
            # best_pred = pred_chunks[max_idx]

            # overlap = compute_overlap(
            #     gt_chunk,
            #     best_pred
            # )

            # # ---------------------------------
            # # SOFT COVERAGE SCORE
            # # ---------------------------------

            # coverage = (
            #     0.7 * best_sim
            #     + 0.3 * overlap
            # )

            total_score += coverage * weight
            total_weight += weight

            details.append({
                "type": u["type"],

                # "gt_chunk": gt_chunk,
                "gt_chunk": gt_chunk,
                "top_matches": [
                    {
                        "pred_chunk": top_pred_chunks[0],
                        "score": round(top_scores[0], 4)
                    }
                ],
                "coverage": round(coverage, 4),
                # "preserved": coverage >= SIM_THRESHOLD

            })

    final_score = (
        total_score / max(total_weight, 1e-8)
    )

    return {
        "layer": layer_name,
        "score": final_score,
        "details": details
    }

def compute_ics_all_layers(
    gt_participant,
    gt_topic,
    gt_meeting,

    pred_participant,
    pred_topic,
    pred_meeting
):
    """
    Compute ICS separately for:
    - participant layer
    - topic layer
    - meeting layer
    """

    # =====================================================
    # PARTICIPANT
    # =====================================================

    participant_result = compute_layer_ics(
        gt_units=gt_participant,
        pred_texts=pred_participant,
        layer_name="participant"
    )

    # =====================================================
    # TOPIC
    # =====================================================

    topic_result = compute_layer_ics(
        gt_units=gt_topic,
        pred_texts=pred_topic,
        layer_name="topic"
    )

    # =====================================================
    # MEETING
    # =====================================================

    meeting_result = compute_layer_ics(
        gt_units=gt_meeting,
        pred_texts=pred_meeting,
        layer_name="meeting"
    )

    # =====================================================
    # FINAL WEIGHTED ICS
    # =====================================================

    p_score = participant_result["score"]
    t_score = topic_result["score"]
    m_score = meeting_result["score"]

    weighted = (
        p_score * INFO_WEIGHTS["participant"]
        + t_score * INFO_WEIGHTS["topic"]
        + m_score * INFO_WEIGHTS["meeting"]
    )

    total_weight = (
        INFO_WEIGHTS["participant"]
        + INFO_WEIGHTS["topic"]
        + INFO_WEIGHTS["meeting"]
    )

    final_score = weighted / total_weight

    return {

        "participant_ics": round(p_score, 4),

        "topic_ics": round(t_score, 4),

        "meeting_ics": round(m_score, 4),

        "overall_ics": round(final_score, 4),

        "participant_details":
            participant_result["details"],

        "topic_details":
            topic_result["details"],

        "meeting_details":
            meeting_result["details"]
    }

# =========================================================
# COMPRESSION RATIO
# =========================================================

def compute_cr(input_text, output_texts):

    in_tokens = token_count(input_text)

    out_tokens = sum(
        token_count(x)
        for x in output_texts
    )

    ratio = out_tokens / max(in_tokens, 1)

    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "compression_ratio": ratio,
        "compression_percent": 1.0 - ratio
    }


# =========================================================
# HALLUCINATION RATE
# =========================================================

def compute_hr(
    transcript,
    generated_texts
):
    """
    Hallucination Rate

    Improvements:
    - split transcript into chunks
    - split generated summaries into claims
    - compare claim ↔ transcript chunk
    - use BOTH:
        1. embedding similarity
        2. lexical overlap
    """

    # =====================================================
    # SPLIT TRANSCRIPT
    # =====================================================

    transcript_chunks = split_sentences(transcript)

    transcript_chunks = [
        normalize(x)
        for x in transcript_chunks
        if normalize(x)
    ]

    if len(transcript_chunks) == 0:
        return 1.0, []

    # =====================================================
    # EMBED TRANSCRIPT CHUNKS
    # =====================================================

    transcript_embs = embedder.encode(
        transcript_chunks,
        convert_to_tensor=True
    )

    # =====================================================
    # SPLIT GENERATED CLAIMS
    # =====================================================

    generated_claims = []

    for t in generated_texts:
        generated_claims.extend(
            split_sentences(t)
        )

    unsupported = 0

    details = []

    # =====================================================
    # CHECK EACH CLAIM
    # =====================================================

    for claim in generated_claims:

        claim_norm = normalize(claim)

        if not claim_norm:
            continue

        # -------------------------------------------------
        # embed claim
        # -------------------------------------------------

        claim_emb = embedder.encode(
            claim_norm,
            convert_to_tensor=True
        )

        # -------------------------------------------------
        # semantic similarity
        # -------------------------------------------------

        sims = util.cos_sim(
            claim_emb,
            transcript_embs
        )[0]

        best_idx = sims.argmax().item()

        best_sim = sims[best_idx].item()

        best_chunk = transcript_chunks[best_idx]

        # -------------------------------------------------
        # lexical overlap
        # -------------------------------------------------

        claim_words = set(claim_norm.split())
        chunk_words = set(best_chunk.split())

        overlap = len(
            claim_words & chunk_words
        ) / max(len(claim_words), 1)

        # -------------------------------------------------
        # hallucination decision
        # -------------------------------------------------

        unsupported_flag = (
            best_sim < SIM_THRESHOLD
            and overlap < 0.30
        )

        if unsupported_flag:
            unsupported += 1

        
        # -------------------------------------------------
        # store details
        # -------------------------------------------------

        details.append({
            "claim": claim,

            "best_transcript_chunk": best_chunk,

            "similarity": round(
                best_sim,
                4
            ),

            "word_overlap": round(
                overlap,
                4
            ),

            "hallucination": unsupported_flag
        })

    # =====================================================
    # FINAL HR
    # =====================================================

    total = max(len(generated_claims), 1)

    hr = unsupported / total

    return hr, details

# =========================================================
# ROUGE-L
# =========================================================

def compute_rouge_l(gt_texts, pred_texts):

    gt = " ".join(gt_texts)
    pred = " ".join(pred_texts)

    if not gt.strip() or not pred.strip():
        return 0.0

    score = rouge.score(
        gt,
        pred
    )

    return score["rougeL"].fmeasure


# =========================================================
# MAIN
# =========================================================

def evaluate_all(
    gt_sample,
    speaker_summary_path,
    topic_summary_path,
    meeting_summary_path
):

    # -------------------------
    # load predictions
    # -------------------------

    speaker_json = load_json(speaker_summary_path)
    topic_json = load_json(topic_summary_path)
    meeting_json = load_json(meeting_summary_path)

    pred = build_prediction_texts(
        speaker_json,
        topic_json,
        meeting_json
    )

    # =====================================================
    # BUILD GT UNITS
    # =====================================================

    gt_participant = flatten_gt_participant(
        gt_sample["participant"]
    )

    gt_topic = flatten_gt_topics(
        gt_sample["topic"]
    )

    gt_meeting = flatten_gt_meeting(
        gt_sample["meeting"]
    )

    gt_all = (
        gt_participant
        + gt_topic
        + gt_meeting
    )

    # =====================================================
    # PRED TEXTS
    # =====================================================

    pred_all = (
        pred["participant"]
        + pred["topic"]
        + pred["meeting"]
    )

    # =====================================================
    # ICS
    # =====================================================

    ics_results = compute_ics_all_layers(

        gt_participant=gt_participant,
        gt_topic=gt_topic,
        gt_meeting=gt_meeting,

        pred_participant=pred["participant"],
        pred_topic=pred["topic"],
        pred_meeting=pred["meeting"]
    )

    ics = ics_results["overall_ics"]

    # =====================================================
    # CR
    # =====================================================

    cr = compute_cr(
        gt_sample["transcript"],
        pred_all
    )

    # =====================================================
    # HR
    # =====================================================

    hr, hr_details = compute_hr(
        gt_sample["transcript"],
        pred_all
    )

    # =====================================================
    # ROUGE-L
    # =====================================================

    gt_texts = [x["text"] for x in gt_all]

    rouge_l = compute_rouge_l(
        gt_texts,
        pred_all
    )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (
        ics
        * (1.0 - hr)
        * (1.0 - cr["compression_ratio"])
    )

    # =====================================================
    # RESULTS
    # =====================================================

    results = {
        "ICS": round(ics, 4),
        "Participant_ICS":
            ics_results["participant_ics"],
        "Topic_ICS":
            ics_results["topic_ics"],
        "Meeting_ICS":
            ics_results["meeting_ics"],
        
        "CR": round(cr["compression_ratio"], 4),
        "SummaryAlrRemove": round(
            cr["compression_percent"],
            4
        ),
        "HR": round(hr, 4),
        "ROUGE_L": round(rouge_l, 4),
        "FinalScore": round(final_score, 4),

        "TokenStats": cr,

        # "ICS_Details": {
        #     "participant":
        #         ics_results["participant_details"],

        #     "topic":
        #         ics_results["topic_details"],

        #     "meeting":
        #         ics_results["meeting_details"]
        # },
        # "HR_Details": hr_details
    }

    return results


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    from sum_data_process import build_meeting_sample

    from config import Config

    cfg = Config()

    BASE_DIR = cfg.base_dir
    MEETING_ID = cfg.meeting_id

    gt_sample = build_meeting_sample(
        BASE_DIR,
        MEETING_ID
    )

    results = evaluate_all(
        gt_sample,
        cfg.speaker_summary,
        cfg.topic_summary,
        cfg.meeting_summary
    )

    print(json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ))