import random
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from transformers import WavLMModel
from speechbrain.inference import EncoderClassifier
from rsmm_utils import is_titanet

def extract_embeddings_from_segments(
    waveform: torch.Tensor,
    classifier: EncDecSpeakerLabelModel | EncoderClassifier,
    speech_segments: list,
    seg_len_sec: float = 3.0,
    sample_rate: int = 16000,
    n_samples: int = 3,
    aggregate: bool = True,
    min_len_ratio: float = 0.5,
    skip_short: bool = False,
) -> torch.Tensor:
    seg_len = int(seg_len_sec * sample_rate)
    min_len = int(min_len_ratio * sample_rate)
    titan = is_titanet(classifier)
    all_embeddings, valid_segments = [], []
    for ts in speech_segments:
        start, end = ts["start"], ts["end"]
        segment = waveform[:, start:end]
        seg_len_actual = segment.shape[-1]

        # Skip too short segments if requested
        if skip_short and seg_len_actual < min_len:
            continue

        # Always pad with zeros if segment shorter than desired length
        if seg_len_actual < seg_len:
            pad_len = seg_len - seg_len_actual
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left

            # Always use constant zero-padding for safety
            segment = torch.nn.functional.pad(segment, (pad_left, pad_right), mode="constant", value=0.0)
            seg_len_actual = segment.shape[-1]

        # Center crop index and random samples
        candidates = [max(0, (seg_len_actual - seg_len) // 2)]
        for _ in range(n_samples - 1):
            candidates.append(random.randint(0, max(0, seg_len_actual - seg_len)))

        sub_embs = []
        for s in candidates:
            crop = segment[:, s : s + seg_len]
            with torch.no_grad():
                if titan:
                    classifier.eval()
                    emb = classifier.forward(
                        input_signal=crop.to(classifier.device),
                        input_signal_length=torch.tensor([crop.shape[-1]]).to(
                            classifier.device
                        ),
                    )
                    if isinstance(emb, tuple):
                        emb = emb[1]
                else:
                    emb = classifier.encode_batch(crop.to(classifier.device))
            sub_embs.append(emb.cpu())

        emb_stack = torch.stack(sub_embs, dim=0)

        # aggregate or keep per-crop embeddings
        if aggregate:
            all_embeddings.append(torch.mean(emb_stack, dim=0))
        else:
            all_embeddings.append(emb_stack)

        valid_segments.append(ts)

    assert len(all_embeddings) > 0, "No embeddings were created"
    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings

def extract_wavlm_embeddings_from_segments(
    waveform: torch.Tensor,
    model: WavLMModel,
    speech_segments: list,
    seg_len_sec: float = 0.5,
    sample_rate: int = 16000,
    n_samples: int = 3,
    aggregate: bool = True,
    min_len_ratio: float = 0.5,
    skip_short: bool = False,
) -> torch.Tensor:
    """
    Extract WavLM embeddings from speech segments, similar to ECAPA/Titanet function style.

    Args:
        waveform: (1, N) waveform tensor
        model: pretrained WavLMModel
        speech_segments: list of {start, end} dicts
        seg_len_sec: segment length in seconds
        sample_rate: waveform sample rate
        n_samples: number of crops per segment
        aggregate: average multiple crops per segment
        min_len_ratio: minimum ratio of segment length before padding
        skip_short: skip too-short segments instead of padding
    """
    seg_len = int(seg_len_sec * sample_rate)
    min_len = int(min_len_ratio * sample_rate)
    device = next(model.parameters()).device

    all_embeddings, valid_segments = [], []

    model.eval()

    for ts in speech_segments:
        start, end = ts["start"], ts["end"]
        segment = waveform[:, start:end]
        seg_len_actual = segment.shape[-1]

        if skip_short and seg_len_actual < min_len:
            continue

        # Pad if too short
        if seg_len_actual < seg_len:
            pad_len = seg_len - seg_len_actual
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left

            use_reflect = (seg_len_actual > 1) and (pad_left < seg_len_actual) and (pad_right < seg_len_actual)
            if use_reflect:
                segment = torch.nn.functional.pad(segment, (pad_left, pad_right), mode="reflect")
            else:
                segment = torch.nn.functional.pad(segment, (pad_left, pad_right), mode="constant", value=0.0)
            seg_len_actual = segment.shape[-1]

        # Choose multiple crops per segment
        candidates = [max(0, (seg_len_actual - seg_len) // 2)]
        for _ in range(n_samples - 1):
            candidates.append(random.randint(0, max(0, seg_len_actual - seg_len)))

        sub_embs = []
        for s in candidates:
            crop = segment[:, s : s + seg_len].to(device)
            crop = crop / (crop.abs().max() + 1e-9)

            with torch.no_grad():
                outputs = model(input_values=crop)
                emb = outputs.last_hidden_state.mean(dim=1)
            sub_embs.append(emb.cpu())

        emb_stack = torch.stack(sub_embs, dim=0)

        if aggregate:
            all_embeddings.append(torch.mean(emb_stack, dim=0))
        else:
            all_embeddings.append(emb_stack)

        valid_segments.append(ts)

    assert len(all_embeddings) > 0, "No embeddings were created"

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings