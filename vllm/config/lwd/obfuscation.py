# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer-obfuscation demo config shared by both LWD roles.

The cloud loads only the runtime artifact (obfuscated vocabulary, no
private permutation); the edge additionally loads the trusted mapping
table and writes the verification report. Paths follow the
mixed-tokenizer build layout: ``runtime/mixed_tokenizer.json`` and
``trusted/byte_permutation.json`` under one ``artifact_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, TypedDict

from .topology import LwdString

LWD_OBF_RUNTIME_ARTIFACT = "runtime/mixed_tokenizer.json"
LWD_OBF_TRUSTED_MAPPING = "trusted/byte_permutation.json"


@with_config(ConfigDict(extra="forbid"))
class LwdObfuscationEntry(TypedDict):
    """User-facing ``lwd_config.obfuscation`` block."""

    artifact_dir: LwdString
    """mixed-tokenizer artifact dir produced by ``mixed_tokenizer build``."""

    report_path: NotRequired[LwdString]
    """Edge-only JSONL verification report; logging only when unset."""


@dataclass(frozen=True)
class LwdObfuscationConfig:
    """Parsed obfuscation block; the feature is on iff the block is present."""

    artifact_dir: str
    report_path: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> LwdObfuscationConfig | None:
        """Parse the obfuscation block; None keeps the feature off."""
        if raw is None:
            return None
        entry = TypeAdapter(LwdObfuscationEntry).validate_python(raw)
        return cls(
            artifact_dir=entry["artifact_dir"],
            report_path=entry.get("report_path"),
        )

    def runtime_artifact_path(self) -> Path:
        return Path(self.artifact_dir) / LWD_OBF_RUNTIME_ARTIFACT

    def trusted_mapping_path(self) -> Path:
        return Path(self.artifact_dir) / LWD_OBF_TRUSTED_MAPPING
