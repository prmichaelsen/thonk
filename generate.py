#!/usr/bin/env python3
"""Generate today's thonk via Replicate google/nano-banana.

Reads manifest.yaml at repo root; computes day_index = (today_utc -
launch_date); picks prompt_rotation[day_index % len(rotation)] as
axis modifier; composes the base prompt; calls Replicate
google/nano-banana; downloads the returned JPEG to
images/thonk-<YYYY-MM-DD>.jpg; appends a scene entry to
manifest.yaml. Idempotent within a UTC day — if today's date already
has a scene, exits 0 without doing anything.

Run: python3 generate.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"
IMAGES = ROOT / "images"
SECRETS = Path.home() / ".acp/projects/reflection/agent/secrets/replicate.api_token"

MODEL_URL = "https://api.replicate.com/v1/models/google/nano-banana/predictions"

BASE_PROMPT_TEMPLATE = (
    'A single emoji-style illustration of a "thonk" face — a '
    "contemplative, considering character with one hand on chin, "
    "furrowed brow, slightly puzzled, thinking deeply. "
    "{AXIS_MODIFIER}. Centered on a {BACKGROUND} background. Round "
    "emoji-style composition, soft shading, expressive eyes. No "
    "text, no caption, no border, no watermark."
)

DEFAULT_BACKGROUND = "soft cream"


def strip_control_chars(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)


def load_token() -> str:
    return SECRETS.read_text().strip()


def create_prediction(prompt: str, token: str) -> dict:
    payload = json.dumps({"input": {"prompt": prompt}}).encode("utf-8")
    req = urllib.request.Request(
        MODEL_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(strip_control_chars(raw))


def poll_prediction(prediction_id: str, token: str) -> dict | None:
    url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    for _ in range(60):
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(strip_control_chars(raw))
        status = data.get("status")
        print(f"  status: {status}", flush=True)
        if status == "succeeded":
            return data
        if status in ("failed", "canceled"):
            print(f"  ERROR: {data.get('error')}", flush=True)
            return None
        time.sleep(3)
    print("  TIMEOUT", flush=True)
    return None


def download(url: str, out_path: Path) -> None:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text()) or {}
    launch_date = manifest["launch_date"]
    if isinstance(launch_date, str):
        launch_date = dt.date.fromisoformat(launch_date)
    rotation = manifest["prompt_rotation"]
    scenes = manifest.get("scenes") or []

    today = dt.datetime.now(dt.timezone.utc).date()
    today_iso = today.isoformat()
    day_index = (today - launch_date).days
    if day_index < 0:
        print(f"today {today_iso} precedes launch {launch_date}", file=sys.stderr)
        return 1

    # Idempotent: if today already has a scene, no-op.
    for scene in scenes:
        if str(scene.get("date")) == today_iso:
            print(f"already generated for {today_iso}; nothing to do")
            return 0

    axis = rotation[day_index % len(rotation)]
    prompt = BASE_PROMPT_TEMPLATE.format(
        AXIS_MODIFIER=axis, BACKGROUND=DEFAULT_BACKGROUND
    )
    print(f"day_index={day_index} date={today_iso}")
    print(f"axis: {axis}")
    print(f"prompt: {prompt}")

    token = load_token()
    print("creating prediction...", flush=True)
    pred = create_prediction(prompt, token)
    pred_id = pred.get("id")
    if not pred_id:
        print(f"FAILED to create prediction: {pred}", file=sys.stderr)
        return 2
    print(f"id: {pred_id}", flush=True)

    final = poll_prediction(pred_id, token)
    if not final:
        return 3

    output = final.get("output")
    output_url = output[0] if isinstance(output, list) else output
    if not output_url:
        print(f"no output url: {final}", file=sys.stderr)
        return 4

    IMAGES.mkdir(parents=True, exist_ok=True)
    img_name = f"thonk-{today_iso}.jpg"
    out_path = IMAGES / img_name
    print(f"downloading to {out_path}...", flush=True)
    download(output_url, out_path)
    print(f"saved: {out_path} ({out_path.stat().st_size} bytes)")

    # Append scene entry to manifest.
    scenes.append(
        {
            "day_index": day_index,
            "date": today_iso,
            "axis": axis,
            "image": img_name,
            "prediction_id": pred_id,
        }
    )
    manifest["scenes"] = scenes
    # Preserve original YAML order by rewriting fully.
    out_text = yaml.dump(
        manifest, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    MANIFEST.write_text(out_text)
    print(f"manifest updated; scenes now {len(scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
