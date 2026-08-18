from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .types import V9_VIRTUAL_TOKEN_COUNT


class TriggerBindingError(ValueError):
    pass


class TriggerPlaceholderError(TriggerBindingError):
    pass


class TriggerConflictError(TriggerBindingError):
    pass


class TriggerTokenizerError(TriggerBindingError):
    pass


class TriggerTokenizerParityError(TriggerTokenizerError):
    pass


class TriggerTruncationError(TriggerBindingError):
    pass


class TriggerAtomicityError(TriggerBindingError):
    pass


@dataclass(frozen=True, slots=True)
class TriggerBindingMetadata:
    raw_text: str
    resolved_text: str
    rendered_text: str
    literal: str
    character_spans: tuple[tuple[int, int], ...]
    token_spans: tuple[tuple[int, int], ...]
    token_indices: tuple[int, ...]
    virtual_token_indices: tuple[int, ...]
    occurrence_indices: tuple[int, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    trigger_mask: tuple[int, ...]
    atomic_token_id: int | None
    lookup_token_id: int | None

    @property
    def occurrence_count(self) -> int:
        return len(self.character_spans)

    @property
    def slot_count(self) -> int:
        return len(self.token_indices)


@dataclass(frozen=True, slots=True)
class TriggerBindingBatch:
    items: tuple[TriggerBindingMetadata, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor
    virtual_token_indices: torch.Tensor
    occurrence_indices: torch.Tensor


def resolve_trigger_literal(raw_text: str, literal: str, placeholder: str = "[trigger]", require_placeholder: bool = True, reject_literal_conflicts: bool = True) -> tuple[str, tuple[tuple[int, int], ...]]:
    if not isinstance(raw_text, str) or not isinstance(literal, str) or not literal or not placeholder:
        raise TriggerPlaceholderError("raw text, placeholder and literal must be valid non-empty strings")
    if require_placeholder and placeholder not in raw_text:
        raise TriggerPlaceholderError(f"caption does not contain required placeholder {placeholder!r}")
    if reject_literal_conflicts and literal in raw_text:
        raise TriggerConflictError("raw caption already contains the literal trigger")
    parts = raw_text.split(placeholder)
    output: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, part in enumerate(parts):
        output.append(part)
        cursor += len(part)
        if index < len(parts) - 1:
            spans.append((cursor, cursor + len(literal)))
            output.append(literal)
            cursor += len(literal)
    return "".join(output), tuple(spans)


def find_literal_spans(text: str, literal: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    while literal:
        index = text.find(literal, start)
        if index < 0:
            break
        spans.append((index, index + len(literal)))
        start = index + len(literal)
    return tuple(spans)


def render_chat_prompt(tokenizer: Any, text: str, add_generation_prompt: bool = True) -> str:
    for content in ([{"type": "text", "text": text}], text):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            if isinstance(rendered, str):
                return rendered
        except (AttributeError, TypeError, ValueError):
            continue
    raise TriggerTokenizerError("tokenizer must support the Ideogram4 Qwen chat template")


def _flat(value: Any, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise TriggerTokenizerError(f"{name} unexpectedly returned a batch")
        value = list(value[0])
    if not isinstance(value, list):
        raise TriggerTokenizerError(f"tokenizer output {name!r} must be a sequence")
    return value


def _tokenize_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return tuple(int(value) for value in _flat(encoded["input_ids"], "input_ids"))


def _tokenize_offsets(
    tokenizer: Any,
    text: str,
    max_length: int | None = None,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    if getattr(tokenizer, "is_fast", None) is not True:
        raise TriggerTokenizerError("trigger binding requires a fast tokenizer")
    kwargs: dict[str, Any] = {
        "add_special_tokens": False,
        "return_offsets_mapping": True,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        if max_length <= 0:
            raise TriggerTruncationError("max_length cannot fit the configured virtual trigger tokens")
        kwargs["max_length"] = int(max_length)
    encoded = tokenizer(text, **kwargs)
    ids = [int(value) for value in _flat(encoded["input_ids"], "input_ids")]
    masks = [int(value) for value in _flat(encoded.get("attention_mask", [1] * len(ids)), "attention_mask")]
    raw_offsets = encoded["offset_mapping"]
    if isinstance(raw_offsets, torch.Tensor):
        raw_offsets = raw_offsets.detach().cpu().tolist()
    if isinstance(raw_offsets, tuple):
        raw_offsets = list(raw_offsets)
    if (
        isinstance(raw_offsets, list)
        and len(raw_offsets) == 1
        and isinstance(raw_offsets[0], list)
        and (not raw_offsets[0] or isinstance(raw_offsets[0][0], (list, tuple)))
    ):
        raw_offsets = raw_offsets[0]
    if not isinstance(raw_offsets, list):
        raise TriggerTokenizerError("offset_mapping must be a sequence")
    offsets = [tuple(map(int, pair)) for pair in raw_offsets]
    if not (len(ids) == len(masks) == len(offsets)):
        raise TriggerTokenizerError("tokenizer output lengths differ")
    return ids, masks, offsets


def _unwrap_huggingface_tokenizer(tokenizer_or_clip: Any) -> Any:
    pending, seen = [tokenizer_or_clip], set()
    while pending:
        value = pending.pop(0)
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if hasattr(value, "get_vocab") and callable(value) and hasattr(value, "apply_chat_template"):
            return value
        for attribute in ("tokenizer", "clip", "clip_l", "qwen3_vl", "qwen3_8b"):
            pending.append(getattr(value, attribute, None))
    raise TriggerTokenizerError("could not locate the connected Hugging Face tokenizer")


def _tokenizer_resource_path(tokenizer: Any) -> Path:
    candidates = [getattr(tokenizer, "name_or_path", None)]
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init_kwargs, Mapping):
        candidates.append(init_kwargs.get("name_or_path"))
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_dir():
            return Path(candidate).expanduser().resolve()
    raise TriggerTokenizerError("connected tokenizer does not expose a local resource directory")


def _special_snapshot(tokenizer: Any) -> tuple[Any, ...]:
    return (
        tuple(sorted((str(k), str(v)) for k, v in dict(getattr(tokenizer, "special_tokens_map", {}) or {}).items())),
        tuple(int(v) for v in getattr(tokenizer, "all_special_ids", ()) or ()),
    )


def validate_fast_slow_tokenizer_parity(slow_tokenizer: Any, fast_tokenizer: Any, rendered_text: str, literal: str | None = None) -> None:
    if slow_tokenizer.get_vocab() != fast_tokenizer.get_vocab():
        raise TriggerTokenizerParityError("fast and slow tokenizer vocabularies differ")
    if _special_snapshot(slow_tokenizer) != _special_snapshot(fast_tokenizer):
        raise TriggerTokenizerParityError("fast and slow tokenizer special-token configuration differs")
    parity_text = rendered_text.replace(literal, "") if literal else rendered_text
    if _tokenize_ids(slow_tokenizer, parity_text) != _tokenize_ids(fast_tokenizer, parity_text):
        raise TriggerTokenizerParityError("fast and slow tokenizer IDs differ")


def create_private_ideogram4_fast_tokenizer(connected_tokenizer_or_clip: Any, parity_text: str = "Ideogram4 tokenizer parity check.", literal: str | None = None, validate_parity: bool = True) -> Any:
    slow = _unwrap_huggingface_tokenizer(connected_tokenizer_or_clip)
    try:
        from transformers import Qwen2TokenizerFast
    except ImportError as exc:
        raise TriggerTokenizerError("transformers.Qwen2TokenizerFast is required") from exc
    try:
        fast = Qwen2TokenizerFast.from_pretrained(str(_tokenizer_resource_path(slow)), local_files_only=True)
    except (OSError, ValueError) as exc:
        raise TriggerTokenizerError("failed to create private local Qwen2TokenizerFast") from exc
    if validate_parity:
        validate_fast_slow_tokenizer_parity(slow, fast, parity_text, literal)
    return fast


def register_atomic_literal(
    tokenizer: Any,
    literal: str,
    vocab_limit: int,
) -> tuple[int, int]:
    lookup_ids = getattr(tokenizer, "_gen2_v9_lookup_token_ids", None)
    if lookup_ids is None:
        lookup_ids = {}
        try:
            setattr(tokenizer, "_gen2_v9_lookup_token_ids", lookup_ids)
        except (AttributeError, TypeError) as exc:
            raise TriggerTokenizerError("private tokenizer cannot retain V9 literal lookup metadata") from exc
    lookup_token_id = lookup_ids.get(literal)
    if lookup_token_id is None:
        source_ids = _tokenize_ids(tokenizer, literal)
        if not source_ids or any(token_id < 0 or token_id >= vocab_limit for token_id in source_ids):
            raise TriggerAtomicityError(
                "literal pre-registration token IDs must remain inside the frozen Qwen embedding table"
            )
        lookup_token_id = int(source_ids[0])
        lookup_ids[literal] = lookup_token_id
    added = tokenizer.add_tokens([literal], special_tokens=True)
    if added not in (0, 1):
        raise TriggerTokenizerError(f"unexpected number of registered tokens: {added}")
    ids = _tokenize_ids(tokenizer, literal)
    if len(ids) != 1:
        raise TriggerAtomicityError(f"literal must map to exactly one token ID, got {list(ids)}")
    atomic_token_id = ids[0]
    if getattr(tokenizer, "unk_token_id", None) == atomic_token_id:
        raise TriggerAtomicityError("literal maps to unknown token ID")
    return atomic_token_id, lookup_token_id


def _overlapping_token(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> int:
    start, end = span
    matches = [index for index, (token_start, token_end) in enumerate(offsets) if token_end > token_start and token_start < end and token_end > start]
    if len(matches) != 1:
        raise TriggerAtomicityError("registered literal must occupy one contextual token")
    index = matches[0]
    if offsets[index][0] > start or offsets[index][1] < end:
        raise TriggerTruncationError("trigger occurrence was partially tokenized")
    return index


def bind_trigger_prompt(
    tokenizer: Any,
    raw_text: str,
    literal: str,
    placeholder: str = "[trigger]",
    vocab_limit: int | None = None,
    max_length: int | None = None,
    stock_literal: bool = False,
) -> TriggerBindingMetadata:
    resolved, resolved_spans = resolve_trigger_literal(raw_text, literal, placeholder)
    rendered = render_chat_prompt(tokenizer, resolved)
    rendered_spans = find_literal_spans(rendered, literal)
    if len(rendered_spans) != len(resolved_spans):
        raise TriggerConflictError(
            "chat template changed or duplicated literal-trigger occurrences; offset mapping is ambiguous"
        )
    if stock_literal:
        ids, attention, _ = _tokenize_offsets(tokenizer, rendered, max_length)
        return TriggerBindingMetadata(raw_text, resolved, rendered, literal, rendered_spans, (), (), (), (), tuple(ids), tuple(attention), tuple(0 for _ in ids), None, None)
    if vocab_limit is None:
        vocab_limit = len(tokenizer.get_vocab())
    atomic_id, lookup_id = register_atomic_literal(tokenizer, literal, vocab_limit)
    expansion_slots = len(rendered_spans) * (V9_VIRTUAL_TOKEN_COUNT - 1)
    tokenizer_max_length = None if max_length is None else max_length - expansion_slots
    if tokenizer_max_length is not None and tokenizer_max_length < len(rendered_spans):
        raise TriggerTruncationError("max_length cannot fit the configured virtual trigger tokens")
    ids, attention, offsets = _tokenize_offsets(tokenizer, rendered, tokenizer_max_length)
    occurrence_positions = [_overlapping_token(offsets, span) for span in rendered_spans]
    expanded_ids: list[int] = []
    expanded_attention: list[int] = []
    trigger_mask: list[int] = []
    virtual_indices: list[int] = []
    occurrence_indices: list[int] = []
    token_spans: list[tuple[int, int]] = []
    occurrence_by_position = {position: index for index, position in enumerate(occurrence_positions)}
    for position, (token_id, mask) in enumerate(zip(ids, attention)):
        occurrence = occurrence_by_position.get(position)
        if occurrence is None:
            expanded_ids.append(token_id)
            expanded_attention.append(mask)
            trigger_mask.append(0)
            virtual_indices.append(-1)
            occurrence_indices.append(-1)
            continue
        if token_id != atomic_id:
            raise TriggerAtomicityError("literal contextual token ID differs from registered atomic ID")
        start = len(expanded_ids)
        for virtual_index in range(V9_VIRTUAL_TOKEN_COUNT):
            expanded_ids.append(atomic_id)
            expanded_attention.append(mask)
            trigger_mask.append(1)
            virtual_indices.append(virtual_index)
            occurrence_indices.append(occurrence)
        token_spans.append((start, start + V9_VIRTUAL_TOKEN_COUNT))
    if max_length is not None and len(expanded_ids) > max_length:
        raise TriggerTruncationError(
            f"four-slot expansion requires {len(expanded_ids)} tokens, exceeding max_length={max_length}"
        )
    token_indices = tuple(index for index, value in enumerate(trigger_mask) if value)
    return TriggerBindingMetadata(
        raw_text, resolved, rendered, literal, rendered_spans, tuple(token_spans), token_indices,
        tuple(virtual_indices[index] for index in token_indices),
        tuple(occurrence_indices[index] for index in token_indices), tuple(expanded_ids),
        tuple(expanded_attention), tuple(trigger_mask), atomic_id, lookup_id,
    )


def bind_trigger_batch(tokenizer: Any, raw_texts: Sequence[str], literal: str, pad_token_id: int | None = None, **kwargs: Any) -> TriggerBindingBatch:
    if not raw_texts:
        raise TriggerBindingError("trigger binding batch must not be empty")
    items = tuple(bind_trigger_prompt(tokenizer, text, literal, **kwargs) for text in raw_texts)
    length = max(len(item.input_ids) for item in items)
    pad = int(pad_token_id if pad_token_id is not None else getattr(tokenizer, "pad_token_id", 0) or 0)
    def rows(name: str, fill: int) -> list[list[int]]:
        return [list(getattr(item, name)) + [fill] * (length - len(getattr(item, name))) for item in items]

    virtual_rows: list[list[int]] = []
    occurrence_rows: list[list[int]] = []
    for item in items:
        virtual = [-1] * len(item.input_ids)
        occurrences = [-1] * len(item.input_ids)
        for token_index, virtual_index, occurrence_index in zip(
            item.token_indices, item.virtual_token_indices, item.occurrence_indices
        ):
            virtual[token_index] = virtual_index
            occurrences[token_index] = occurrence_index
        virtual_rows.append(virtual + [-1] * (length - len(virtual)))
        occurrence_rows.append(occurrences + [-1] * (length - len(occurrences)))
    return TriggerBindingBatch(
        items,
        torch.tensor(rows("input_ids", pad), dtype=torch.long),
        torch.tensor(rows("attention_mask", 0), dtype=torch.long),
        torch.tensor(rows("trigger_mask", 0), dtype=torch.bool),
        torch.tensor(virtual_rows, dtype=torch.long),
        torch.tensor(occurrence_rows, dtype=torch.long),
    )


__all__ = [
    "TriggerAtomicityError", "TriggerBindingBatch", "TriggerBindingError", "TriggerBindingMetadata",
    "TriggerConflictError", "TriggerPlaceholderError", "TriggerTokenizerError", "TriggerTokenizerParityError",
    "TriggerTruncationError", "bind_trigger_batch", "bind_trigger_prompt",
    "create_private_ideogram4_fast_tokenizer", "find_literal_spans", "register_atomic_literal",
    "render_chat_prompt", "resolve_trigger_literal", "validate_fast_slow_tokenizer_parity",
]
