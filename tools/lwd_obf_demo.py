# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client driver of the LWD tokenizer-obfuscation demo.

Sends one completion to the edge API and checks the edge-side obfuscation
report against the HTTP response: the verifier's restored text (cloud
obfuscated detokenize + trusted edge restore) must equal the native
detokenization of the same token ids, and the HTTP text must equal the
report's skip-special-tokens view of them.

Prerequisites (see docs/design/lwd_obf_demo.puml for the full flow):
  - both sides started with ``VLLM_ASCEND_LWD_DISABLE_DOWN=1`` (token ids
    return via c2e, no DOWN hidden transfer);
  - both sides carry ``lwd_config.obfuscation`` (same artifact_dir; the
    edge additionally writes ``report_path``);
  - the artifact dir comes from ``mixed_tokenizer build`` and the
    ``mixed_tokenizer`` package is installed in both environments.

Example:
  python tools/lwd_obf_demo.py --endpoint http://127.0.0.1:8000 \
      --model Qwen/Qwen3-8B --report /tmp/lwd_obf_report.jsonl \
      --prompt "用一句话介绍边云协同推理" --max-tokens 64
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def _post_completion(endpoint: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body["choices"][0]["text"]


def _new_report_records(report: Path, lines_before: int) -> list[dict]:
    lines = report.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[lines_before:]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompt", default="Introduce edge-cloud LLM inference.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    if not args.report.is_file():
        # The edge writes the file lazily on the first finished request.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.touch()
    lines_before = len(args.report.read_text(encoding="utf-8").splitlines())

    print(f"[demo] prompt: {args.prompt!r}")
    http_text = _post_completion(
        args.endpoint, args.model, args.prompt, args.max_tokens
    )
    print(f"[demo] path-1 HTTP text: {http_text!r}")

    # The report line lands on the engine thread right at request finish;
    # it may trail the HTTP response by a moment.
    records: list[dict] = []
    for _ in range(100):
        time.sleep(0.2)
        records = _new_report_records(args.report, lines_before)
        if records:
            break
    if not records:
        print("[demo] FAIL: no verification record appeared in the report")
        return 1

    record = records[-1]
    native = record["text_native"]
    restored = record["text_restored"]
    print(f"[demo] tokens={record['num_tokens']} request={record['request_id']}")
    print(f"[demo] path-1 native detokenize : {native!r}")
    print(f"[demo] path-2 restored from obf : {restored!r}")

    failures: list[str] = []
    if not record["pass"]:
        failures.append("restored text differs from native detokenization")
    if http_text != record["text_http"]:
        failures.append(
            f"HTTP text differs from the skip-special view: {http_text!r} vs "
            f"{record['text_http']!r}"
        )
    if failures:
        print(f"[demo] FAIL: {'; '.join(failures)}")
        return 1
    print("[demo] PASS: obfuscation/de-obfuscation roundtrip matches the native path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
