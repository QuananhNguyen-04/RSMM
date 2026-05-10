import json
from collections import defaultdict
import os
from typing import List, Dict

import dotenv
from groq import Groq
import re

from config import Config

# ======================
# CONFIG
# ======================

client = Groq(api_key=dotenv.get_key(dotenv.find_dotenv(), "GROQ_API"))
MIN_TOKENS = 3

# ======================
# LOAD DATA
# ======================

def load_transcript(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======================
# PREPROCESSING
# ======================

def is_valid_text(text: str) -> bool:
    return len(text.strip().split()) >= MIN_TOKENS

def clean_text(text: str) -> str:
    return " ".join(text.strip().split())

def merge_segments(segments: List[Dict]) -> List[Dict]:
    merged = []
    prev = None

    for seg in segments:
        text = clean_text(seg["text"])
        if not is_valid_text(text):
            continue

        if prev and seg["speaker"] == prev["speaker"]:
            prev["text"] += " " + text
            prev["end"] = seg["end"]
        else:
            if prev:
                merged.append(prev)
            prev = {
                "speaker": seg["speaker"],
                "text": text,
                "start": seg["start"],
                "end": seg["end"]
            }

    if prev:
        merged.append(prev)

    return merged

# ======================
# FIXED GROUP BY SPEAKER
# ======================

def group_by_speaker(segments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    KEEP FULL STRUCTURE
    """
    speaker_map = defaultdict(list)
    for seg in segments:
        speaker_map[seg["speaker"]].append(seg)
    return speaker_map

# ======================
# UTIL
# ======================

def clean_llm_output(text):
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

# ======================
# INFO EXTRACTION
# ======================

def extract_info_llm(client, segments):
    system_prompt = """
Extract informative evidence units.

Use ONLY transcript content.

Do NOT add new ideas or rewrite content.

Preserve original wording and order whenever possible.

Keep important:
- decisions
- requirements
- constraints
- findings
- proposals

Remove only filler and repetition.

Prefer high information coverage.

Return ONLY valid JSON, NO comments, NO explanations:
{
  "info":[
    {
      "text":"...",
      "start":0.0,
      "end":0.0
    }
  ]
}
"""

    context = "\n".join(
        f"[{s['start']}-{s['end']}] {s['text']}"
        for s in segments
    )
    print(context)
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input:\n{context}"}
            ],
            temperature=0.1,
        )   
        print(response)
        output = clean_llm_output(response.choices[0].message.content)
        parsed = json.loads(output)

        return parsed.get("info", [])

    except Exception as e:
        print(f"[INFO ERROR] {e}")
        return []

# ======================
# SPEAKER SUMMARY
# ======================

def summarize_info_llm(client, info_list):
    system_prompt = """
Summarize speaker evidence using ONLY input information.

Do NOT add new ideas, conclusions, interpretations, or categories.

Preserve original wording, phrases, terminology, and sentence order whenever possible.

Prefer extraction and light compression over paraphrasing.

Keep important:
- decisions
- requirements
- constraints
- findings
- proposals
- responsibilities

Remove only filler and repetition.

Prefer high information coverage over aggressive compression.

Keep concise when possible.
Do NOT output lists
Do NOT output bullet points
Do NOT output nested JSON objects

Return ONLY valid JSON, NO comments, NO explanations:
{
  "summary":"single paragraph text"
}
"""

    context = "\n".join(f"- {i['text']}" for i in info_list)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input:\n{context}"}
            ],
            temperature=0.4,
        )

        output = clean_llm_output(response.choices[0].message.content)
        parsed = json.loads(output)

        return parsed.get("summary", "")

    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")
        return ""


def group_into_topics(
    speaker_results,
    max_topics=6,
):
    """
    Split topics using temporal gaps.

    A new topic starts when:
        next.start - prev.end > gap_threshold

    Hard capped by max_topics.
    """

    # ======================
    # FLATTEN
    # ======================
    gap_threshold = 6 * 15 / max_topics
    all_items = []

    for sp in speaker_results:
        for item in sp["info"]:
            all_items.append({
                "speaker": sp["id"],
                "text": item["text"],
                "start": item["start"],
                "end": item["end"]
            })

    if not all_items:
        return []

    # ======================
    # SORT
    # ======================

    all_items.sort(key=lambda x: x["start"])

    # ======================
    # GROUP
    # ======================

    topics = []
    current_topic = [all_items[0]]

    for item in all_items[1:]:

        prev = current_topic[-1]

        gap = item["start"] - prev["end"]

        should_split = (
            gap > gap_threshold
            and len(topics) < (max_topics - 1)
        )

        if should_split:
            topics.append(current_topic)
            current_topic = [item]
        else:
            current_topic.append(item)

    if current_topic:
        topics.append(current_topic)

    # ======================
    # FORMAT
    # ======================

    formatted_topics = []

    for idx, topic_items in enumerate(topics):

        formatted_topics.append({
            "topic_id": idx,
            "start": topic_items[0]["start"],
            "end": topic_items[-1]["end"],
            "items": topic_items
        })

    return formatted_topics

def summarize_topic_llm(client, topic):
    system_prompt = """
Compress one meeting topic into a dense factual summary. Use ONLY input information.

Do NOT add new ideas, conclusions, interpretations, or categories.

Preserve original wording, phrases, terminology, and sentence order whenever possible.

Prefer extraction and light compression over paraphrasing.

Keep as many important points as possible:
- decisions
- requirements
- constraints
- findings
- proposals

Remove only filler and repetition.

Prefer high information coverage over aggressive compression.

Return ONLY valid JSON, NO comments, NO explanations, summary contains single paragraph text, NO nested JSON, no bullet points:
{
  "evidence": [
    {
      "speaker": "...",
      "text": "...",
      "start": 0.0,
      "end": 0.0
    }
  ]
  "summary": "single paragraph text",
}
"""

    context = "\n".join(
        f"[{i['start']}-{i['end']}] [{i['speaker']}] {i['text']}"
        for i in topic["items"]
    )
    # print(context)
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input:\n{context}"}
        ],
        temperature=0.2,
    )

    output = clean_llm_output(response.choices[0].message.content)
    print(output)
    return json.loads(output)

def build_topics(client, speaker_results, max_topics=4):
    topics = group_into_topics(speaker_results, max_topics)

    topic_outputs = []

    for t in topics:
        summary = summarize_topic_llm(client, t)

        topic_outputs.append({
            "topic_id": t["topic_id"],
            "summary": summary.get("summary", ""),
            "evidence": summary.get("evidence", [])[:5]
        })

    return topic_outputs


def save_topics(input_path, topics):
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    base_name = base_name.replace("final_transcriptions_","")

    save_dir = "topic_summary"

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir,f"topic_summarization_{base_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(topics,f,ensure_ascii=False,indent=2)

    print(f"Saved topics to: {save_path}")
# ======================
# PROCESS SPEAKER
# ======================

def process_speaker(client, speaker_id: str, segments: List[Dict]):
    info = extract_info_llm(client, segments)
    print("info", info)
    summary = summarize_info_llm(client, info)

    return {
        "id": speaker_id,
        "info": info,
        "summary": summary
    }

# ======================
# PIPELINE
# ======================

def run_pipeline(path: str):
    raw_segments = load_transcript(path)

    merged_segments = merge_segments(raw_segments)

    speaker_map = group_by_speaker(merged_segments)

    results = []

    for speaker_id, segments in speaker_map.items():
        print(f"Processing {speaker_id}")
        result = process_speaker(client, speaker_id, segments)
        results.append(result)

    return results

def save_results(path, results):
    base_name = os.path.splitext(os.path.basename(path))[0]
    base_name = base_name.replace("final_transcriptions_","")
    save_dir = "speaker_sum"
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"speaker_summarization_{base_name}.json"

    save_path = os.path.join(save_dir,save_name)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {save_path}")

# ======================
# MEETING SUMMARY
# ======================

def summarize_final_llm(client, evidence_units):
    system_prompt = """
Task:
Produce a single compact meeting summary using ONLY the provided topic summaries.

You are NOT allowed to restructure the output.
Rules:
- You MUST use ONLY words, phrases, or closely matching fragments from the input topic summaries.
- Use ONLY information from topic summaries
- Do NOT invent new structure or categories
- Do NOT create bullet points or lists
- Preserve key factual content
- Compress aggressively but keep meaning

Content focus:
Include ONLY:
- major decisions
- key requirements
- critical findings
- important constraints

Style:
- Single coherent paragraph only
- Dense factual writing
- Minimal wording
- No headers, no labels, no enumeration

Length:
- Maximum 150 words

Output format (STRICT):
Return ONLY valid JSON, NO comments, NO explanations:

{
  "summary": "single paragraph text here"
}

No other keys are allowed.
No explanation.
No markdown.
No extra text.
"""
    context_parts = []

    for t in evidence_units:
        topic_text = f"Topic {t['topic_id']}:\n"
        topic_text += f"Summary: {t['summary']}\n"

        for ev in t["evidence"]:
            topic_text += f"- [{ev['speaker']}] {ev['text']}\n"

        context_parts.append(topic_text)

    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input:\n{context}"}
        ],
        temperature=0.2,
    )

    output = clean_llm_output(response.choices[0].message.content)
    print(output)
    return json.loads(output)

def save_meeting_summary(input_path, result):
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    base_name = base_name.replace("final_transcriptions_","")
    save_dir = "meeting_summary"

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir,f"meeting_summarization_{base_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)

    print(f"Saved to: {save_path}")

# ======================
# MAIN
# ======================

if __name__ == "__main__":
    cfg = Config()

    path = cfg.pred_path

    speaker_results = run_pipeline(path)
    save_results(path, speaker_results)

    topics = build_topics(client, speaker_results)

    save_topics(path, topics)

    meeting_result = summarize_final_llm(client, topics)
    save_meeting_summary(path, meeting_result)

# # ======================
# # CHUNKING (SAFE SPLIT)
# # ======================

# def chunk_texts(texts: List[str], chunk_size=CHUNK_SIZE) -> List[List[str]]:
#     return [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

# def build_context(texts: List[str]) -> str:
#     return "\n".join(f"- {t}" for t in texts)

# # ======================
# # OPTIONAL: ENTITY EXTRACTION
# # ======================

# def extract_entities_simple(texts: List[str]) -> List[str]:
#     # lightweight placeholder (can replace with spaCy later)
#     entities = set()
#     for t in texts:
#         for word in t.split():
#             if word.istitle():  # naive heuristic
#                 entities.add(word)
#     return list(entities)

# # ======================
# # LLM CALL PLACEHOLDER
# # ======================

# def call_llm(prompt: str) -> str:
#     """
#     Replace this with your actual LLM call (Groq / OpenAI / etc.)
#     Must return JSON string.
#     """
#     raise NotImplementedError


# # ======================
# # INFO EXTRACTION (MERGEABLE)
# # ======================

# INFO_PROMPT = """
# You are given utterances from ONE speaker.

# Select the most representative sentences.

# Rules:
# - MUST use exact or minimally cleaned sentences
# - DO NOT paraphrase
# - Remove filler if needed
# - Avoid redundancy
# - Prefer concrete information

# Output JSON:
# {
#   "info": ["...", "..."]
# }
# """

# def extract_info_llm(client, speaker_id, texts, chunk_size=8):
#     system_prompt = """
# Task: Extract representative evidence sentences from a single speaker.

# You may:
# - Select exact sentences or minimally cleaned versions
# - Remove filler words (uh, yeah, okay)
# - Merge adjacent sentences if they form one idea

# You must:
# - NOT paraphrase
# - NOT invent new text
# - NOT change meaning
# - Avoid redundancy
# - Output ONLY JSON

# Format:
# {
#   "info": ["...", "..."]
# }
# """

#     # Few-shot example (important for stability)
#     user_prompt = """
# Input:
# - this is where we talk properties, materials, user interface
# - we had a couple of changes in our plans
# - we couldn't use teletext, it wasn't going to be control for everything
# """

#     assistant_prompt = """
# {
#   "info": [
#     "this is where we talk properties, materials, user interface",
#     "we had a couple of changes in our plans",
#     "we couldn't use teletext, it wasn't going to be control for everything"
#   ]
# }
# """
#     def build_context(texts):
#         return "\n".join(f"- {t}" for t in texts)

#     def call_llm(context):
#         request_prompt = f"Input:\n{context}\n"

#         response = client.chat.completions.create(
#             model="meta-llama/llama-4-scout-17b-16e-instruct",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#                 {"role": "assistant", "content": assistant_prompt},
#                 {"role": "user", "content": request_prompt},
#             ],
#             temperature=0.3,
#         )

#         output = response.choices[0].message.content
#         return json.loads(output).get("info", [])

#     # ===== CASE 1: small → 1 call =====
#     if len(texts) <= 15:
#         return call_llm(build_context(texts))

#     # ===== CASE 2: large → split into 2 =====
#     mid = len(texts) // 2
#     part1 = texts[:mid]
#     part2 = texts[mid:]

#     info1 = call_llm(build_context(part1))
#     info2 = call_llm(build_context(part2))

#     # merge + deduplicate
#     seen = set()
#     merged = []

#     for item in info1 + info2:
#         key = item.lower()
#         if key not in seen:
#             seen.add(key)
#             merged.append(item)

#     return merged[:6]

# def extract_info(texts: List[str]) -> List[str]:
#     chunks = chunk_texts(texts)
#     collected_info = []

#     for chunk in chunks:
#         context = build_context(chunk)
#         prompt = INFO_PROMPT + "\n\n" + context

#         output = call_llm(prompt)
#         parsed = json.loads(output)

#         collected_info.extend(parsed.get("info", []))

#     # Deduplicate (simple)
#     unique_info = []
#     seen = set()

#     for item in collected_info:
#         key = item.lower()
#         if key not in seen:
#             seen.add(key)
#             unique_info.append(item)

#     return unique_info[:MAX_INFO_ITEMS]

# # ======================
# # SUMMARY FROM INFO
# # ======================

# SUMMARY_PROMPT = """
# You are given evidence sentences from a speaker.

# Write a concise summary (2-4 sentences).

# Rules:
# - ONLY use information from the evidence
# - DO NOT add new facts
# - Stay faithful

# Output JSON:
# {
#   "summary": "..."
# }
# """

# def summarize_info(info: List[str]) -> str:
#     context = "\n".join(f"- {i}" for i in info)
#     prompt = SUMMARY_PROMPT + "\n\n" + context

#     output = call_llm(prompt)
#     parsed = json.loads(output)

#     return parsed["summary"]

# def summarize_info_llm(client, speaker_id, info_list):
#     system_prompt = """
# Task: Summarize a speaker using ONLY given evidence sentences.

# You may:
# - Rephrase for clarity

# You must:
# - NOT add new information
# - NOT introduce new facts
# - Stay strictly grounded in the evidence
# - Keep it concise (2-4 sentences)
# - Output ONLY JSON

# Format:
# {
#   "summary": "..."
# }
# """

#     user_prompt = """
# Input:
# - this is where we talk properties, materials, user interface
# - we had a couple of changes in our plans
# - we couldn't use teletext, it wasn't going to be control for everything
# """

#     assistant_prompt = """
# {
#   "summary": "The speaker discusses topics related to properties, materials, and user interface. They mention changes in plans from previous discussions and highlight a constraint where teletext cannot be used."
# }
# """

#     context = "\n".join(f"- {i}" for i in info_list)
#     request_prompt = f"Input:\n{context}\n"

#     try:
#         response = client.chat.completions.create(
#             model="meta-llama/llama-4-scout-17b-16e-instruct",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#                 {"role": "assistant", "content": assistant_prompt},
#                 {"role": "user", "content": request_prompt},
#             ],
#             temperature=0.4,
#         )

#         output = response.choices[0].message.content
#         parsed = json.loads(output)

#         return parsed["summary"]

#     except Exception as e:
#         print(f"[SUMMARY ERROR] {e}")
#         return ""
# # ======================
# # MAIN SPEAKER PIPELINE
# # ======================

# def process_speaker(client, speaker_id: str, texts: List[str], use_entities=False):
#     info = extract_info_llm(client, speaker_id, texts)
#     summary = summarize_info_llm(client, speaker_id, info)

#     result = {
#         "id": speaker_id,
#         "info": info,
#         "summary": summary
#     }

#     if use_entities:
#         result["entities"] = extract_entities_simple(texts)

#     return result

# # ======================
# # FULL PIPELINE
# # ======================

# def run_pipeline(path: str, use_entities=False):
#     raw_segments = load_transcript(path)

#     merged_segments = merge_segments(raw_segments)

#     speaker_map = group_by_speaker(merged_segments)

#     results = []

#     for speaker_id, texts in speaker_map.items():
#         print(f"Processing {speaker_id} ")
#         print(texts)
#         result = process_speaker(client, speaker_id, texts, False)
#         results.append(result)
#     print(results)
#     return results

# def save_results(path, results):
#     base_name = os.path.splitext(os.path.basename(path))[0]
#     save_dir = os.path.join("speaker_sum", base_name)
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(save_dir, "speaker_sum.json")

#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     print(f"Saved to: {save_path}")
    
# def chunk_segments_bounded(segments, max_chunks=4):
#     n = len(segments)
#     chunk_size = max(1, n // max_chunks)

#     chunks = []
#     for i in range(0, n, chunk_size):
#         chunks.append(segments[i:i+chunk_size])

#     return chunks[:max_chunks]  # hard cap
# def summarize_chunk_llm(client, segments):
#     system_prompt = """
# Task: Summarize a meeting segment and extract supporting evidence.

# You must:
# - Write a clear, informative summary (NOT overly abstract)
# - Keep important details (decisions, findings, constraints)

# Evidence:
# - MUST be exact excerpt from input
# - MUST include speaker
# - DO NOT paraphrase evidence

# Output ONLY JSON:

# {
#   "summary": "...",
#   "evidence": [
#     {"speaker": "...", "text": "..."}
#   ]
# }
# """

#     user_prompt = """
# Input:
# [Speaker 1] we did a market study with 100 users
# [Speaker 2] it seems users prefer a fancy design
# """

#     assistant_prompt = """
# {
#   "summary": "Findings from a market study involving 100 users, and preference in appealing designs.",
#   "evidence": [
#     {"speaker": "Speaker 1", "text": "a market study with 100 users"},
#     {"speaker": "Speaker 2", "text": "users prefer a fancy design"}
#   ]
# }
# """

#     context = build_context(segments)
#     request_prompt = f"Input:\n{context}\n"

#     try:
#         response = client.chat.completions.create(
#             model="meta-llama/llama-4-scout-17b-16e-instruct",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#                 {"role": "assistant", "content": assistant_prompt},
#                 {"role": "user", "content": request_prompt},
#             ],
#             temperature=0.4,
#         )

#         return json.loads(response.choices[0].message.content)

#     except Exception as e:
#         print(f"[CHUNK ERROR] {e}")
#         return {"summary": "", "evidence": []}

# def summarize_meeting(client, segments, max_chunks=4):
#     # chunks = chunk_segments_bounded(segments, max_chunks)

#     # chunk_outputs = []

#     # for i, chunk in enumerate(chunks):
#     #     print(f"Processing chunk {i+1}/{len(chunks)}...")
#     #     result = summarize_chunk_llm(client, chunk)
#     #     chunk_outputs.append(result)
#     # print(chunk_outputs)
#     chunk_outputs = [
#     {
#         'summary': "The team discussed a new remote control design. Market research showed demand for a stylish, innovative, and easy-to-use device. They chose a flip design with a unique shape reflecting the company’s image, without voice recognition. Battery options considered were rechargeable, solar, and kinetic.", 
#         'evidence': [
#             {'speaker': 'Speaker 4', 'text': 'the general findings from that was in market trends the most important aspects for remote controls were people want a fancy People want a fancy look and feel rather than the current functional look and feel of remote controls.'}, 
#             {'speaker': 'Speaker 3', 'text': "We have decided on leaving out the voice recognition. We've decided on there being a flip design and a different shape from what's normal, we were thinking a shell, but something along the sides, just a different shape from what's normal."}, 
#             {'speaker': 'Speaker 0', 'text': "there's actually no rechargeable option available so we I saw the the standard AA and AAA which we thought were a bit too bulky at the moment."}, 
#             {'speaker': 'Speaker 0', 'text': "I think the next one is the best anyway, the kinetic charging, which is like you get it in wristwatches, yeah, and you don't even notice it."}
#         ]}, 
#     {
#         'summary': "The discussion revolves around design considerations for a remote control, including material choices, interface design, and features. The team debates the use of standard batteries versus rechargeable options, with Speaker 0 suggesting that using standard batteries might detract from the product's attractiveness. They also discuss the possibility of adding a kinetic feature and the importance of considering user preferences. The team explores various interface design options, including silicon PCB boards, touch screens, and LCD screens. They weigh the pros and cons of each approach, considering factors such as cost, durability, and user experience. The conversation also touches on the idea of customizing circuits for different types of buttons and the potential for users to change the look of the remote control.", 
#         'evidence': [
#             {'speaker': 'Speaker 0', 'text': "if you had something using the standard batteries and the cellular charging, I don't think it would. Well, you know how long the standard double A's would last in or triple A's would last. It would just detach from the attractiveness of the whole feature."}, 
#             {'speaker': 'Speaker 3', 'text': "Okay, can we add in an attachment to closing feature? Okay. Can we think about that? Because if we're doing the kinetic thing. Yeah. Shouldn't we"}, 
#             {'speaker': 'Speaker 4', 'text': 'do some more research on that before we add it in?'}, 
#             {'speaker': 'Speaker 0', 'text': 'The thing about it is that they can be as big or as small as you want them to be because you can, to be, because you can print a circuit board like that.'}, 
#             {'speaker': 'Speaker 2', 'text': "Yeah and second question is like mobile you can change the cover the skin or whatever so in this case if you're looking at like a customer can change the color like from green parrot green to chili red or something yeah is that feature available in like titanium or it's like only specific to yeah in"}, 
#             {'speaker': 'Speaker 0', 'text': "titanium I don't I don't think it would be available at all really the just it would well you could make it available in the titanium it was just it would be so expensive to buy a new case for it because of the expense of how much titanium is it's light and strong but I think it should be left for aircraft design rather than for it doesn't"}
#             ]
#         }, 
#         {'summary': "The meeting discussed the design and features of a new remote control. Key findings from a market study include a preference for trendy designs with a focus on user experience. Competitor analysis showed monotonous designs with too many cluttered buttons and inconsistencies. Proposed features include a GUI interface, voice recognition, and a possible 'find the remote' feature. Decisions were made on battery type and design elements.", 'evidence': [{'speaker': 'Speaker 2', 'text': "we'll have to share something which I did here. First thing is, basically on design, we just took the input from the previous meeting, especially from the marketing and industrial design, to check on the customer needs and feasibility. Second is we check into competitors. The picture here shows what are the standard models offered by competitors here. So you generally see there's not much of variety and like marketing team said, people need trendy, they're both of black and white. board of black and white so you generally see rectangular shape very monotonous kind of designs here and second thing is there's too much of confusion here no particular remote is standard"}, {'speaker': 'Speaker 2', 'text': 'So the findings are too many cluttered buttons. The repetition of certain patterns, which I already explained, examples of volume and channel control buttons. All are confusing and inconsistent. Okay, we had a latest finding of voice recognition. There was a mail which mentions that our division has developed a new speech recognition feature. We have to check into the financial feasibility whether we can incorporate this at a low cost.'}, {'speaker': 'Speaker 3', 'text': "Okay, so we've got the battery. The inside components are pretty standardised across the board,"}]}, {'summary': 'The meeting discussed product design and features. Key decisions include a flip-top design with an LCD display on top and rubberized buttons on the bottom. The body will be plastic with a rubberized cover, and colors will be based on fruit and vegetables. The design should be different from existing products, with options for various colors and designs. A detachable body and interchangeable case covers were also suggested. The group discussed but did not decide on voice recognition or a beeper for locating the device if lost.', 'evidence': [{'speaker': 'Speaker 4', 'text': "for the decisions that we've made, kinetic charging, the watch type batteries, LCD display on the top side of the flip top. The top side of the flip top, rubberised buttons on the bottom side, we're going to use fruit and vegetable colour to cover the case itself as plastic."}, {'speaker': 'Speaker 3', 'text': "It's just different. It's just different from everything else."}, {'speaker': 'Speaker 2', 'text': "Fine, we're talking of voice recognition also, because we have not addressed the issue of how to locate a remote control if it's lost."}, {'speaker': 'Speaker 4', 'text': "So, are you looking at voice recognition? We're just a hostage issue with that, but it's a good idea, we just need to check on the cost. Or maybe like volume"}, {'speaker': 'Speaker 2', 'text': "we're pretty going in a clear direction."}]
#     }
# ]
#     final_result = summarize_final_llm(client, chunk_outputs)

#     return final_result

# def trim_chunk_outputs(chunk_outputs, max_evidence=3):
#     trimmed = []

#     for c in chunk_outputs:
#         trimmed.append({
#             "summary": c.get("summary", ""),
#             "evidence": c.get("evidence", [])[:max_evidence]
#         })

#     return trimmed

# import re

# def clean_llm_output(text):
#     if not text:
#         return ""

#     text = text.strip()

#     # remove ```json ... ```
#     if text.startswith("```"):
#         text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
#         text = re.sub(r"\n?```$", "", text)

#     return text.strip()

# def summarize_final_llm(client, chunk_outputs):
#     system_prompt = """
# Task: Combine multiple meeting summaries into one final summary.

# You must:
# - Produce a clear, informative final summary
# - Avoid redundancy
# - Preserve key decisions and findings

# Evidence:
# - MUST come from provided evidence only
# - MUST keep speaker attribution

# Output ONLY JSON:

# {
#   "summary": "...",
#   "evidence": [...]
# }
# """

#     chunk_outputs = trim_chunk_outputs(chunk_outputs, max_evidence=3)
#     context = json.dumps(chunk_outputs, ensure_ascii=False, indent=2)
#     print(context)
#     response = client.chat.completions.create(
#         model="meta-llama/llama-4-scout-17b-16e-instruct",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": context},
#         ],
#         temperature=0.4,
#     )
#     print(response)
#     output = response.choices[0].message.content
#     cleaned = clean_llm_output(output)
#     # print(response.choices[0].message.content ,json.loads(response.choices[0].message.content))
#     print(cleaned, json.loads(cleaned))
#     return json.loads(cleaned)

#     # except Exception as e:
#     #     print(f"[FINAL ERROR] {e}")
#     #     return {"summary": "", "evidence": []}
    
# def save_meeting_summary(input_path, result):
#     base_name = os.path.splitext(os.path.basename(input_path))[0]
#     save_dir = os.path.join("meeting_summary", base_name)
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(save_dir, "overall_summary.json")

#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"Saved to: {save_path}")
    
# if __name__ == "__main__":

#     filepath = "./transcriptions/final_transcriptions1.json"
#     result = run_pipeline(filepath)
    
#     save_results(filepath, result)


#     path = "./transcriptions/final_transcriptions2.json"

#     segments = load_transcript(path)

#     result = summarize_meeting(client, segments)

#     save_meeting_summary(path, result)

# ======================
# MAIN
# ======================

# if __name__ == "__main__":
#     cfg = Config()

#     filepath = cfg.get("paths", "transcriptions", "input_1")
#     result = run_pipeline(filepath)
#     save_results(filepath, result)

#     path = cfg.get("paths", "transcriptions", "input_1")
#     # filepath = "./transcriptions/final_transcriptions1.json"
#     # result = run_pipeline(filepath)
#     # save_results(filepath, result)

#     # path = "./transcriptions/final_transcriptions2.json"
#     # segments = load_transcript(path)

#     meeting_summary = summarize_meeting(client, result)
#     save_meeting_summary(path, result)