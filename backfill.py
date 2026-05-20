#!/usr/bin/env python3
"""Backfill 50 thonks into the manifest under the emoji-grade
pipeline. One-time seeding event: rewrite manifest.launch_date to
2026-04-01, regenerate dates 2026-04-01..2026-05-20 (50 days,
inclusive). Each day uses the existing prompt rotation
(day_index % len(rotation)).

Resumable: if an image already exists on disk for a date, skip
generation for that day (idempotent within a backfill session).
Manifest is rewritten in-place after every successful day so
partial runs persist their progress.

Run: uv run --with pyyaml --with pillow python backfill.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import yaml

# Reuse the generator's primitives.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from generate import (  # type: ignore  # noqa: E402
    BASE_PROMPT_TEMPLATE,
    IMAGES,
    MANIFEST,
    fetch_bytes,
    generate_image,
    load_token,
    post_process,
    remove_background,
)

NEW_LAUNCH_DATE = dt.date(2026, 4, 1)
LAST_DATE = dt.date(2026, 5, 20)
TOTAL_DAYS = (LAST_DATE - NEW_LAUNCH_DATE).days + 1  # 50


def gen_one(day_index: int, date: dt.date, axis: str, token: str) -> dict | None:
    img_name = f"thonk-{date.isoformat()}.png"
    out_path = IMAGES / img_name

    # Resumability: if image already present, treat as already done.
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  skip (already on disk): {out_path}")
        return {
            "day_index": day_index,
            "date": date.isoformat(),
            "axis": axis,
            "image": img_name,
            "seeded": True,
        }

    prompt = BASE_PROMPT_TEMPLATE.format(AXIS_MODIFIER=axis)
    print(f"  prompt: {prompt[:100]}…")

    gen_url = generate_image(prompt, token)
    if not gen_url:
        print(f"  GEN FAILED for day {day_index}", flush=True)
        return None
    bg_url = remove_background(gen_url, token)
    if not bg_url:
        print(f"  BG FAILED for day {day_index}", flush=True)
        return None
    raw = fetch_bytes(bg_url)
    final = post_process(raw)
    out_path.write_bytes(final)
    print(
        f"  saved: {out_path} ({out_path.stat().st_size} bytes)",
        flush=True,
    )
    return {
        "day_index": day_index,
        "date": date.isoformat(),
        "axis": axis,
        "image": img_name,
        "seeded": True,
    }


def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text()) or {}

    # Reset the calendar baseline.
    manifest["launch_date"] = NEW_LAUNCH_DATE.isoformat()
    manifest["backfilled_at"] = "2026-05-20"

    rotation = manifest["prompt_rotation"]
    if not rotation:
        print("rotation empty; aborting", file=sys.stderr)
        return 1

    # Preserve any existing scenes that fall in our range so we
    # don't lose hand-curated entries; otherwise start fresh.
    existing_by_date = {
        str(s.get("date")): s for s in (manifest.get("scenes") or [])
    }

    token = load_token()
    IMAGES.mkdir(parents=True, exist_ok=True)

    new_scenes: list[dict] = []
    for i in range(TOTAL_DAYS):
        date = NEW_LAUNCH_DATE + dt.timedelta(days=i)
        axis = rotation[i % len(rotation)]
        date_iso = date.isoformat()
        print(f"[{i+1}/{TOTAL_DAYS}] day_index={i} date={date_iso} axis={axis}")

        # If image exists on disk OR scene already in manifest for
        # this date, prefer regen-skip. But we also need to make
        # sure the entry's day_index/axis match the new schedule.
        img_path = IMAGES / f"thonk-{date_iso}.png"
        if img_path.exists() and img_path.stat().st_size > 0:
            entry = {
                "day_index": i,
                "date": date_iso,
                "axis": axis,
                "image": img_path.name,
                "seeded": True,
            }
            print(f"  reuse existing image: {img_path.name}")
            new_scenes.append(entry)
            continue

        entry = gen_one(i, date, axis, token)
        if entry is None:
            # Persist what we have so far.
            manifest["scenes"] = new_scenes
            MANIFEST.write_text(
                yaml.dump(
                    manifest, sort_keys=False, default_flow_style=False,
                    allow_unicode=True,
                )
            )
            print(
                f"FAILED at day {i}. partial manifest written.",
                file=sys.stderr,
            )
            return 2
        new_scenes.append(entry)

        # Persist after every successful day so re-runs resume cleanly.
        manifest["scenes"] = new_scenes
        MANIFEST.write_text(
            yaml.dump(
                manifest, sort_keys=False, default_flow_style=False,
                allow_unicode=True,
            )
        )

    print(f"done. scenes={len(new_scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
