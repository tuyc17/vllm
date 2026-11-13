# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer-level obfuscation demo on the LWD edge-cloud link.

Cloud side: ``LwdCloudObfDetokenizer`` holds the obfuscated tokenizer
snapshot (runtime artifact only, no private permutation) and turns each
step's sampled token ids into PUA-carrier obfuscated text that rides the
c2e notify. Token ids stay native, so the id space is shared with the
edge and the cloud never learns the plaintext byte values.

Edge side (trusted): ``LwdEdgeObfVerifier`` restores the carrier stream
(carrier -> mixed bytes -> permutation inverse -> incremental UTF-8) and
at request finish compares it with the native detokenization of the same
delivered token ids. The verdict is logged and appended to the optional
JSONL report. Only the edge reads the trusted table, matching the
private-mapping boundary of the mixed-tokenizer deployment.

The ``mixed_tokenizer`` package (vllm-mixed-tokenizer-demo) is an
optional dependency: it is imported once at load time and a missing
install fails fast with a setup hint instead of failing per request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.config.lwd import LwdObfuscationConfig
from vllm.lwd_logger import lwd_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_LWD_OBF_SETUP_HINT = (
    "[LWD][obf] the obfuscation demo requires the mixed_tokenizer package "
    "(vllm-mixed-tokenizer-demo, install it into this environment) or drop "
    "lwd_config.obfuscation from the config"
)


def _lwd_load_runtime_artifact(config: LwdObfuscationConfig) -> Any:
    """Load the cloud-side MixedTokenizer snapshot, fail fast on problems."""
    path = config.runtime_artifact_path()
    if not path.is_file():
        raise ValueError(f"[LWD][obf] runtime artifact not found: {path}")
    try:
        from mixed_tokenizer.runtime import MixedTokenizer
    except ImportError as exc:
        raise RuntimeError(_LWD_OBF_SETUP_HINT) from exc
    return MixedTokenizer.from_file(path)


class LwdCloudObfDetokenizer:
    """Cloud-side obfuscated tokenizer: token ids -> carrier text."""

    def __init__(self, tokenizer: Any, bytes_to_carrier: Any) -> None:
        self._tokenizer = tokenizer
        self._bytes_to_carrier = bytes_to_carrier

    @classmethod
    def from_config(cls, config: LwdObfuscationConfig) -> LwdCloudObfDetokenizer:
        tokenizer = _lwd_load_runtime_artifact(config)
        from mixed_tokenizer.transport import bytes_to_carrier

        return cls(tokenizer, bytes_to_carrier)

    def obf_carrier_text(self, token_ids: list[int]) -> str:
        """Untokenize one request's new token ids into obfuscated text.

        Raises:
            ValueError: an id has no byte representation (vocabulary
                mismatch); the caller surfaces it without killing the
                engine.
        """
        data = b"".join(
            self._tokenizer.token_bytes(token_id) for token_id in token_ids
        )
        return self._bytes_to_carrier(data)


@dataclass(frozen=True)
class _LwdMixedCodec:
    """Trusted mixed-tokenizer callables bound once at load time."""

    permutation: Any
    """BytePermutation holding the private pi / pi^-1 table."""

    decoder_cls: Any
    """TrustedOutputDecoder class (one incremental instance per request)."""

    carrier_to_bytes: Any
    """Carrier text -> mixed bytes (the inverse of bytes_to_carrier)."""


class _LwdObfStream:
    """Per-request accumulation: native ids plus the incremental restore."""

    __slots__ = ("ids", "decoder")

    def __init__(self, decoder: Any) -> None:
        self.ids: list[int] = []
        self.decoder = decoder


class LwdEdgeObfVerifier:
    """Trusted edge restoration and the two-path equality check.

    ``feed`` accumulates one c2e row (token ids + obfuscated carrier
    text); ``verify`` runs at request finish: it flushes the incremental
    restoration, decodes the same ids natively and compares both texts.
    """

    def __init__(
        self,
        codec: _LwdMixedCodec,
        report_path: Path | None,
        model_config: ModelConfig,
    ) -> None:
        self._codec = codec
        self._report_path = report_path
        self._model_config = model_config
        self._streams: dict[str, _LwdObfStream] = {}
        self._native: Any = None

    @classmethod
    def from_config(
        cls, config: LwdObfuscationConfig, model_config: ModelConfig
    ) -> LwdEdgeObfVerifier:
        """Load the runtime artifact (public binding) and the trusted table.

        The binding checks (profile, digest, commitment) are all performed
        by ``load_trusted_mapping``; a mismatched pair refuses to start.
        """
        runtime = _lwd_load_runtime_artifact(config)
        trusted_path = config.trusted_mapping_path()
        if not trusted_path.is_file():
            raise ValueError(f"[LWD][obf] trusted mapping not found: {trusted_path}")
        try:
            from mixed_tokenizer.build import load_trusted_mapping
            from mixed_tokenizer.frontend import TrustedOutputDecoder
            from mixed_tokenizer.transport import carrier_to_bytes
        except ImportError as exc:
            raise RuntimeError(_LWD_OBF_SETUP_HINT) from exc
        permutation = load_trusted_mapping(
            trusted_path,
            profile_id=runtime.profile_id,
            artifact_digest=runtime.artifact_digest,
            permutation_commitment=runtime.permutation_commitment,
        )
        codec = _LwdMixedCodec(permutation, TrustedOutputDecoder, carrier_to_bytes)
        report_path = Path(config.report_path) if config.report_path else None
        if report_path is not None:
            # Fail at startup on an unwritable report location instead of
            # failing the engine thread on the first finished request.
            report_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(codec, report_path, model_config)

    def feed(self, request_id: str, token_ids: list[int], obf_text: str) -> None:
        """Accumulate one delivered c2e row of a request."""
        stream = self._streams.get(request_id)
        if stream is None:
            stream = _LwdObfStream(self._codec.decoder_cls(self._codec.permutation))
            self._streams[request_id] = stream
            if token_ids and not obf_text:
                lwd_logger.warning(
                    "[LWD][obf-verify] req=%s tokens arrived without obfuscated "
                    "text: cloud obfuscation is off or misaligned, verification "
                    "will fail",
                    request_id,
                )
        stream.ids.extend(token_ids)
        if obf_text:
            stream.decoder.feed_bytes(self._codec.carrier_to_bytes(obf_text))

    def verify(self, request_id: str) -> dict[str, Any] | None:
        """Finish the stream, compare both paths, log and report.

        Returns the record (also appended to the JSONL report when
        configured); None when the request was never fed (aborted before
        any delivery).
        """
        stream = self._streams.pop(request_id, None)
        if stream is None:
            return None
        text_restored = stream.decoder.finish()
        token_ids = stream.ids
        native = self._native_tokenizer()
        # The hard assertion compares full byte streams (special tokens
        # included); text_http mirrors the frontend detokenizer's default
        # skip_special_tokens=True output for the client-side sanity check.
        text_native = native.decode(token_ids, skip_special_tokens=False)
        text_http = native.decode(token_ids, skip_special_tokens=True)
        passed = text_native == text_restored
        record = {
            "request_id": request_id,
            "pass": passed,
            "num_tokens": len(token_ids),
            "text_native": text_native,
            "text_http": text_http,
            "text_restored": text_restored,
        }
        lwd_logger.info(
            "[LWD][obf-verify] req=%s %s num_tokens=%d native_chars=%d "
            "restored_chars=%d",
            request_id,
            "PASS" if passed else "FAIL",
            len(token_ids),
            len(text_native),
            len(text_restored),
        )
        if not passed:
            lwd_logger.error(
                "[LWD][obf-verify] req=%s mismatch native=%r restored=%r",
                request_id,
                text_native[:160],
                text_restored[:160],
            )
        if self._report_path is not None:
            with self._report_path.open("a", encoding="utf-8") as report:
                report.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _native_tokenizer(self) -> Any:
        """Lazily load the native tokenizer backend (tokenizers-level decode
        inserts no spaces, matching the byte-stream comparison)."""
        if self._native is None:
            from vllm.tokenizers.registry import cached_tokenizer_from_config

            native = cached_tokenizer_from_config(self._model_config)
            self._native = getattr(native, "_tokenizer", native)
        return self._native
