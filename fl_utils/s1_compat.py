from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import tiktoken
import torch
from loguru import logger


FISH_TIKTOKEN_PATTERN = "|".join(
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class S1Tokenizer:
    def __init__(self, tokenizer_path: Path, special_tokens: dict[str, int]):
        mergeable_ranks = {}
        for line in tokenizer_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            token, rank = line.split()
            if token != "=":
                mergeable_ranks[base64.b64decode(token)] = int(rank)

        self.all_special_tokens_with_ids = special_tokens
        self.semantic_id_to_token_id = {
            int(match.group(1)): token_id
            for token, token_id in special_tokens.items()
            if (match := re.fullmatch(r"<\|semantic:(\d+)\|>", token))
        }
        if not self.semantic_id_to_token_id:
            raise ValueError("Fish Speech checkpoint contains no semantic tokens")

        self.semantic_begin_id = min(self.semantic_id_to_token_id.values())
        self.semantic_end_id = max(self.semantic_id_to_token_id.values())
        self.tkt_model = tiktoken.Encoding(
            name=tokenizer_path.stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        special_tokens = json.loads((checkpoint_path / "special_tokens.json").read_text(encoding="utf-8"))
        return cls(checkpoint_path / "tokenizer.tiktoken", special_tokens)

    @property
    def vocab_size(self):
        return self.tkt_model.n_vocab

    def get_token_id(self, token: str) -> int:
        return self.all_special_tokens_with_ids[token]

    def encode(self, text: str, allowed_special=True, **_kwargs) -> list[int]:
        if allowed_special is True:
            allowed_special = self.tkt_model.special_tokens_set
        elif allowed_special is False:
            allowed_special = set()

        chunks = [text[i : i + TIKTOKEN_MAX_ENCODE_CHARS] for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)]
        return sum(
            self.tkt_model.encode_batch(chunks, allowed_special=allowed_special, disallowed_special=set()),
            start=[],
        )

    def decode(self, tokens, **_kwargs) -> str:
        return self.tkt_model.decode(tokens)


def configure_s1_tokenizer(model, checkpoint_path: str | Path):
    tokenizer = S1Tokenizer.from_pretrained(checkpoint_path)
    expected_codebooks = model.config.codebook_size
    if len(tokenizer.semantic_id_to_token_id) != expected_codebooks:
        raise ValueError(
            f"Fish Speech checkpoint has {len(tokenizer.semantic_id_to_token_id)} semantic tokens, expected {expected_codebooks}"
        )

    model.tokenizer = tokenizer
    model.config.semantic_begin_id = tokenizer.semantic_begin_id
    model.config.semantic_end_id = tokenizer.semantic_end_id
    logger.info(
        f"Loaded OpenAudio S1 tokenizer with semantic range "
        f"{tokenizer.semantic_begin_id}-{tokenizer.semantic_end_id}"
    )
    return tokenizer


def generate_long_s1(
    *,
    model,
    device,
    decode_one_token,
    text,
    num_samples=1,
    max_new_tokens=0,
    top_p=0.9,
    top_k=30,
    repetition_penalty=1.1,
    temperature=1.0,
    compile=False,
    iterative_prompt=True,
    chunk_length=512,
    prompt_text=None,
    prompt_tokens=None,
    progress_callback=None,
):
    from fish_speech.content_sequence import ContentSequence, TextPart, VQPart
    from fish_speech.models.text2semantic.inference import GenerateResponse, generate

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]
    if use_prompt and len(prompt_text) != len(prompt_tokens):
        raise ValueError("Prompt text and tokens must have the same length")

    sequence = ContentSequence(modality="interleave")
    if use_prompt:
        for prompt, codes in zip(prompt_text, prompt_tokens):
            sequence.append([TextPart(text=prompt), VQPart(codes=codes.cpu())], add_end=True, speaker=0)
    sequence.append(TextPart(text=text), add_end=False, speaker=0)

    encoded, audio_masks, audio_parts = sequence.encode_for_inference(
        model.tokenizer, num_codebooks=model.config.num_codebooks
    )
    if encoded.size(1) > model.config.max_seq_len - 2048:
        raise ValueError(f"Prompt is too long: {encoded.size(1)} > {model.config.max_seq_len - 2048}")

    encoded = encoded.to(device=device)
    prompt_length = encoded.size(1)
    model_size = sum(parameter.numel() for parameter in model.parameters())

    for _ in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated = generate(
            model=model,
            prompt=encoded,
            max_new_tokens=max_new_tokens,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            decode_one_token=decode_one_token,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - started
        codes = generated[1:, prompt_length:-1].clone()
        if not (codes >= 0).all():
            raise RuntimeError("Fish Speech generated negative audio codes")
        logger.info(
            f"Generated {codes.shape[1]} tokens in {elapsed:.2f} seconds, "
            f"{model_size * codes.shape[1] / max(elapsed, 1e-9) / 1e9:.2f} GB/s"
        )
        yield GenerateResponse(action="sample", codes=codes, text=text)

    yield GenerateResponse(action="next")
