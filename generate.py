#!/usr/bin/env python3
"""Generate today's thonk via Replicate google/nano-banana, then
strip the background via Replicate 851-labs/background-remover, and
post-process locally to a 256x256 RGBA PNG — emoji-grade for
Slack/Discord.

Pipeline (Approach A — stays cloud-side):
  1. Read manifest.yaml; pick today's axis from prompt_rotation.
  2. Compose the prompt with white-background helper text.
  3. Call google/nano-banana — get an RGB image URL.
  4. Call 851-labs/background-remover with that URL — get an
     RGBA PNG URL.
  5. Download the bg-removed PNG.
  6. PIL post-process: ensure RGBA, center-crop to square, resize
     to 256x256, optimize.
  7. Save to images/thonk-<YYYY-MM-DD>.png.
  8. Append scene to manifest.yaml.

Idempotent within a UTC day: if today already has a scene, exits 0
without doing anything.

Run: uv run --with pyyaml --with pillow python generate.py
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.yaml"
IMAGES = ROOT / "images"
SECRETS = Path.home() / ".acp/projects/reflection/agent/secrets/replicate.api_token"

GEN_MODEL_URL = "https://api.replicate.com/v1/models/google/nano-banana/predictions"
BG_REMOVE_META_URL = (
    "https://api.replicate.com/v1/models/851-labs/background-remover"
)
PREDICTIONS_URL = "https://api.replicate.com/v1/predictions"

BASE_PROMPT_TEMPLATE = (
    'A single emoji-style illustration of a "thonk" face — a '
    "contemplative, considering character with one hand on chin, "
    "furrowed brow, slightly puzzled, thinking deeply. "
    "{AXIS_MODIFIER}. Centered subject, isolated figure, on a "
    "clean white background, no shadow, no border, no watermark, "
    "no text, no caption. Round emoji-style composition, soft "
    "shading, expressive eyes."
)

TARGET_SIZE = 256


def strip_control_chars(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)


def load_token() -> str:
    return SECRETS.read_text().strip()


def _post_json(url: str, token: str, body: dict) -> dict:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
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


def _output_url(final: dict) -> str | None:
    output = final.get("output")
    if isinstance(output, list):
        return output[0] if output else None
    if isinstance(output, str):
        return output
    return None


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def generate_image(prompt: str, token: str) -> str | None:
    """Call nano-banana; return URL of generated image."""
    print("creating nano-banana prediction...", flush=True)
    pred = _post_json(GEN_MODEL_URL, token, {"input": {"prompt": prompt}})
    pred_id = pred.get("id")
    if not pred_id:
        print(f"FAILED to create gen prediction: {pred}", file=sys.stderr)
        return None
    print(f"  id: {pred_id}", flush=True)
    final = poll_prediction(pred_id, token)
    if not final:
        return None
    return _output_url(final)


def _get_latest_version(meta_url: str, token: str) -> str:
    # Replicate's meta endpoint 403s on the default urllib User-Agent;
    # use a generic UA to bypass the gate.
    req = urllib.request.Request(
        meta_url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "thonk-worker/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(strip_control_chars(raw))
    return data["latest_version"]["id"]


def remove_background(image_url: str, token: str) -> str | None:
    """Call 851-labs/background-remover; return URL of RGBA PNG.
    Community models require version-based prediction endpoint."""
    print("fetching bg-remover latest version...", flush=True)
    version = _get_latest_version(BG_REMOVE_META_URL, token)
    print(f"  version: {version}", flush=True)
    print("creating bg-removal prediction...", flush=True)
    pred = _post_json(
        PREDICTIONS_URL,
        token,
        {"version": version, "input": {"image": image_url}},
    )
    pred_id = pred.get("id")
    if not pred_id:
        print(f"FAILED to create bg prediction: {pred}", file=sys.stderr)
        return None
    print(f"  id: {pred_id}", flush=True)
    final = poll_prediction(pred_id, token)
    if not final:
        return None
    return _output_url(final)


def post_process(png_bytes: bytes) -> bytes:
    """Ensure RGBA, center-crop to square, resize to TARGET_SIZE,
    optimize. Returns PNG bytes."""
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.size[0] != TARGET_SIZE:
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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

    # FORCE flag for regeneration (overwrite today's scene).
    force = "--force" in sys.argv

    if not force:
        for scene in scenes:
            if str(scene.get("date")) == today_iso:
                print(f"already generated for {today_iso}; nothing to do")
                return 0

    axis = rotation[day_index % len(rotation)]
    prompt = BASE_PROMPT_TEMPLATE.format(AXIS_MODIFIER=axis)
    print(f"day_index={day_index} date={today_iso}")
    print(f"axis: {axis}")
    print(f"prompt: {prompt}")

    token = load_token()
    gen_url = generate_image(prompt, token)
    if not gen_url:
        return 2
    print(f"  gen output: {gen_url}", flush=True)

    bg_url = remove_background(gen_url, token)
    if not bg_url:
        return 3
    print(f"  bg-removed output: {bg_url}", flush=True)

    print("downloading bg-removed png...", flush=True)
    raw_png = fetch_bytes(bg_url)
    print(f"  fetched {len(raw_png)} bytes", flush=True)

    print("post-processing (RGBA → square → 256x256)...", flush=True)
    final_png = post_process(raw_png)

    IMAGES.mkdir(parents=True, exist_ok=True)
    img_name = f"thonk-{today_iso}.png"
    out_path = IMAGES / img_name
    out_path.write_bytes(final_png)
    print(f"saved: {out_path} ({out_path.stat().st_size} bytes)")

    # Sanity check the alpha channel survived.
    check = Image.open(out_path)
    print(f"  mode={check.mode} size={check.size}")
    assert check.mode == "RGBA", f"expected RGBA, got {check.mode}"

    # Update manifest: replace today's scene if --force, else append.
    new_scene = {
        "day_index": day_index,
        "date": today_iso,
        "axis": axis,
        "image": img_name,
    }
    replaced = False
    if force:
        for i, scene in enumerate(scenes):
            if str(scene.get("date")) == today_iso:
                scenes[i] = new_scene
                replaced = True
                break
    if not replaced:
        scenes.append(new_scene)
    manifest["scenes"] = scenes
    out_text = yaml.dump(
        manifest, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    MANIFEST.write_text(out_text)
    print(f"manifest updated; scenes now {len(scenes)} (replaced={replaced})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
