import os
import glob
import re
import xml.etree.ElementTree as ET

from collections import Counter

# =========================
# 1. WORDS → TRANSCRIPT
# =========================

def parse_words_file(path):
    tree = ET.parse(path)
    root = tree.getroot()

    words = []
    current_sentence = []

    for elem in root.iter():
        tag = elem.tag.split('}')[-1]

        if tag == "w":
            text = (elem.text or "").strip()

            if elem.attrib.get("punc") == "true":
                # attach punctuation to previous word
                if current_sentence:
                    current_sentence[-1] += text
            else:
                if text:
                    current_sentence.append(text)

        elif tag in ["sil", "pause"]:
            continue

    return " ".join(current_sentence)


def build_transcript(words_dir, meeting_id):
    pattern = os.path.join(words_dir, f"{meeting_id}*.words.xml")
    files = sorted(glob.glob(pattern))

    texts = []
    for f in files:
        texts.append(parse_words_file(f))

    return " ".join(texts)


# =========================
# 2. PARTICIPANT SUMMARY → INFO UNITS
# =========================

def parse_participant_summary(path):
    tree = ET.parse(path)
    root = tree.getroot()

    units = []

    for elem in root.iter():
        tag = elem.tag.split('}')[-1]

        if tag == "sent":
            text = (elem.text or "").strip()
            if text and not text.isdigit():  # remove "1.", "2."
                units.append(text)

    return units


# =========================
# 3. ABSTRACTIVE SUMMARY (OPTIONAL)
# =========================

def parse_abstractive(path):
    tree = ET.parse(path)
    root = tree.getroot()

    texts = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]

        if tag in ["sentence", "sent"]:
            t = (elem.text or "").strip()
            if t:
                texts.append(t)

    return " ".join(texts)


def build_abstractive(abssumm_dir, meeting_id):
    path = os.path.join(abssumm_dir, f"{meeting_id}.abssumm.xml")
    if not os.path.exists(path):
        return None

    return parse_abstractive(path)

def build_word_index(words_dir, meeting_id):
    pattern = os.path.join(words_dir, f"{meeting_id}*.words.xml")
    files = glob.glob(pattern)

    word_index = {}  # speaker → list of (id, text)
    def get_nite_id(elem):
        for k, v in elem.attrib.items():
            if k.endswith("id"):
                return v
        return None
    for f in files:
        fname = os.path.basename(f)
        speaker = fname.split('.')[1]   # EN2001a.A.words.xml → A
        tree = ET.parse(f)
        root = tree.getroot()

        words = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]

            if tag == "w":
                # print(elem.attrib)
                wid = get_nite_id(elem)
                text = (elem.text or "").strip()
                if text:
                    words.append((wid, text))

        word_index[speaker] = words

    return word_index

import re
def analyze_topic_distinctness(topics):
    def wid(x):
        return int(x.split("words")[-1])

    # -------------------------
    # 1. normalize + merge per topic
    # -------------------------
    def merge(spans):
        spans = [(spk, wid(s), wid(e)) for spk, s, e in spans]
        spans.sort(key=lambda x: (x[0], x[1]))

        merged = []
        i = 0
        while i < len(spans):
            spk, s, e = spans[i]
            j = i + 1

            while j < len(spans):
                spk2, s2, e2 = spans[j]
                if spk2 != spk:
                    break
                if s2 <= e + 1:
                    e = max(e, e2)
                    j += 1
                else:
                    break

            merged.append((spk, s, e))
            i = j

        return merged

    topic_spans = [merge(t["spans"]) for t in topics]

    # -------------------------
    # 2. convert to span sets
    # -------------------------
    def expand(spans):
        s = set()
        for spk, st, en in spans:
            for i in range(st, en + 1):
                s.add((spk, i))
        return s

    topic_sets = [expand(s) for s in topic_spans]

    # -------------------------
    # 3. distinctness analysis
    # -------------------------
    results = []

    for i, tset in enumerate(topic_sets):
        others = set().union(*[topic_sets[j] for j in range(len(topic_sets)) if j != i])

        unique = tset - others
        shared = tset & others

        total = len(tset)

        results.append({
            "topic_id": i,
            "total_units": total,
            "unique_units": len(unique),
            "shared_units": len(shared),
            "distinct_ratio": len(unique) / max(total, 1)
        })

    # -------------------------
    # 4. global summary
    # -------------------------
    avg_distinct = sum(r["distinct_ratio"] for r in results) / max(len(results), 1)

    return {
        "per_topic": results,
        "avg_distinct_ratio": avg_distinct
    }

def parse_topics(topics_dir, meeting_id):
    path = os.path.join(topics_dir, f"{meeting_id}.topic.xml")
    if not os.path.exists(path):
        return []

    tree = ET.parse(path)
    root = tree.getroot()

    topics = []

    def extract_topic(t):
        desc = t.attrib.get("other_description", "").strip()

        spans = []
        for child in t.findall("./{*}child"):  # ONLY direct children
            href = child.attrib.get("href")
            if not href:
                continue

            file_part, id_part = href.split("#")
            speaker = file_part.split(".")[1]

            ids = re.findall(r'id\((.*?)\)', id_part)
            if len(ids) == 2:
                spans.append((speaker, ids[0], ids[1]))

        # recursively extract subtopics
        subtopics = []
        for sub in t.findall("./{*}topic"):
            subtopics.extend(extract_topic(sub))

        # keep only meaningful topics
        if desc or spans:
            return [{
                "description": desc,
                "spans": spans
            }] + subtopics
        else:
            return subtopics

    for t in root.findall("./{*}topic"):
        topics.extend(extract_topic(t))

    return topics

# =========================
# CLEAN LAYER BUILDERS
# =========================

def build_participant_layer(summ_dir, meeting_id):
    pattern = os.path.join(summ_dir, f"{meeting_id}*.summ.xml")
    files = glob.glob(pattern)

    result = {}

    for f in files:
        fname = os.path.basename(f)
        speaker = fname.split('.')[1]   # IS1003b.A.summ.xml → A

        units = parse_participant_summary(f)

        result[speaker] = units

    return result

def build_topic_layer_simple(word_index, topics):

    def build_id_map(words):
        return {wid: i for i, (wid, _) in enumerate(words)}

    # precompute id → index
    index_map = {
        spk: build_id_map(word_index[spk])
        for spk in word_index
    }
    # print(index_map)
    results = []

    for t in topics:
        text_parts = []
        unit_count = 0  # number of spans successfully extracted

        for spk, s_id, e_id in t["spans"]:
            words = word_index.get(spk, [])
            idx_map = index_map.get(spk, {})

            if s_id not in idx_map or e_id not in idx_map:
                continue

            s_idx = idx_map[s_id]
            e_idx = idx_map[e_id]

            if s_idx > e_idx:
                continue

            segment_words = [w for _, w in words[s_idx:e_idx+1]]
            segment_text = " ".join(segment_words).strip()

            if segment_text:
                text_parts.append(segment_text)
                unit_count += 1

        final_text = "".join(text_parts)

        results.append({
            "description": t["description"],
            "text": final_text,
            "units": unit_count,          # number of spans used
            "length": len(final_text.split())  # total words
        })
    
    return results

def build_meeting_layer(abstractive_text):
    if not abstractive_text:
        return []

    # split into sentences (simple but enough)
    sentences = re.split(r'[.!?]+', abstractive_text)

    return [s.strip() for s in sentences if s.strip()]

def build_meeting_sample(base_dir, meeting_id):
    """
    Return: meeting_id, transcript, participant_layer, topic_layer, meeting_layer
    """
    words_dir = os.path.join(base_dir, "words")
    summ_dir = os.path.join(base_dir, "participantSummaries")
    abs_dir = os.path.join(base_dir, "abstractive")
    topics_dir = os.path.join(base_dir, "topics")

    # base
    transcript = build_transcript(words_dir, meeting_id)
    abstractive = build_abstractive(abs_dir, meeting_id)

    # structures
    word_index = build_word_index(words_dir, meeting_id)
    
    topics = parse_topics(topics_dir, meeting_id)
    # print(topics)
    analysis= analyze_topic_distinctness(topics)
    # print(analysis)
    # ===== 3 SEPARATE LAYERS =====
    participant_layer = build_participant_layer(summ_dir, meeting_id)
    topic_layer = build_topic_layer_simple(word_index, topics)
    # analysis = analyze_topic_sentences(topic_layer)
    meeting_layer = build_meeting_layer(abstractive)

    return {
        "meeting_id": meeting_id,
        "transcript": transcript,

        # Layer 1
        "participant": participant_layer,

        # Layer 2
        "topic": topic_layer,

        # Layer 3
        "meeting": meeting_layer
    }
# =========================
# 5. USAGE
# =========================
if __name__ == "__main__":
    # base_dir = "D:/Downloads/ami_public_manual_1.6.2"

    # meeting_id = "IS1003b"   # change this
    # print("hello world")
    from config import Config

    cfg = Config()

    base_dir = cfg.base_dir
    meeting_id = cfg.meeting_id
    
    # print(f"{base_dir}/{meeting_id}")
    sample = build_meeting_sample(base_dir, meeting_id)

    print("Participant_layer: ", sample["participant"])
    print("Topic: ", sample["topic"])
    print("Meeting: ", sample["meeting"])


    # print("Transcript length:", len(sample["transcript"].split()))
    # print("Info units:", len(sample["info_units"]))
    # print("Sample units:", sample["info_units"][:5])
    # print("Abstractive:", sample["meeting_summary_gt"][:100])