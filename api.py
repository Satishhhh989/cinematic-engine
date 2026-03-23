"""
FastAPI wrapper for the Mathematical Cinematic Motion Intelligence Engine.

Features:
  • Firebase Authentication (Bearer token)
  • Daily upload limit (3/day per user via Firestore)
  • Video duration validation (≤ 15 s)
  • Static file serving for outputs

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import firebase_admin
from firebase_admin import auth, credentials, firestore
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from main import run_engine

log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MAX_DURATION_SEC = 15.0
DAILY_UPLOAD_LIMIT = 3

app = FastAPI(title="Cinematic Motion Intelligence API")

# ── CORS ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Firebase & Firestore ─────────────────────────────────────────────

_db = None


@app.on_event("startup")
def _startup() -> None:
    global _db

    # Create directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize Firebase Admin SDK from environment variable
    cred_json = os.getenv("FIREBASE_CREDENTIALS")
    if not cred_json:
        raise RuntimeError("FIREBASE_CREDENTIALS environment variable not set")

    cred_dict = json.loads(cred_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    log.info("Firebase initialized")


# ── Static file serving ──────────────────────────────────────────────

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ── Helpers ───────────────────────────────────────────────────────────


def _verify_token(authorization: str | None) -> dict:
    """Validate the Bearer token and return the decoded payload."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return decoded


def _check_and_increment_limit(uid: str) -> None:
    """Enforce daily upload limit via Firestore.  Raises 403 if exceeded."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc_ref = _db.collection("users").document(uid)
    doc = doc_ref.get()

    if doc.exists:
        data = doc.to_dict()
        last_date = data.get("last_upload_date", "")
        uploads = data.get("uploads_today", 0)

        if last_date != today:
            uploads = 0

        if uploads >= DAILY_UPLOAD_LIMIT:
            raise HTTPException(
                status_code=403,
                detail={"error": "Daily upload limit reached (3 per day)"},
            )

        doc_ref.update({
            "uploads_today": uploads + 1,
            "last_upload_date": today,
        })
    else:
        doc_ref.set({
            "uid": uid,
            "uploads_today": 1,
            "last_upload_date": today,
        })


def _cleanup(*paths: Path) -> None:
    """Silently remove files if they exist."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# ── Endpoints ─────────────────────────────────────────────────────────


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """Accept a video upload, authenticate, enforce limits, and run the engine."""

    # ── 1. Firebase auth ──────────────────────────────────────────
    decoded = _verify_token(authorization)
    uid = decoded["uid"]

    # ── 2. Daily upload limit ─────────────────────────────────────
    _check_and_increment_limit(uid)

    # ── 3. Save upload ────────────────────────────────────────────
    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}.mp4"
    output_path = OUTPUT_DIR / f"{job_id}_out.mp4"

    try:
        contents = await file.read()
        input_path.write_bytes(contents)
    except Exception as exc:
        log.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # ── 4. Validate duration ──────────────────────────────────────
    try:
        cap = cv2.VideoCapture(str(input_path))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            _cleanup(input_path)
            raise HTTPException(status_code=400, detail="Invalid video file")

        duration = frame_count / fps
        if duration > MAX_DURATION_SEC:
            _cleanup(input_path)
            return {"error": "Video must be 15 seconds or less"}
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Duration check failed: %s", exc)
        _cleanup(input_path)
        raise HTTPException(status_code=400, detail="Could not read video metadata")

    # ── 5. Run engine ─────────────────────────────────────────────
    try:
        run_engine(str(input_path), str(output_path))
    except Exception as exc:
        log.error("Engine failed: %s", exc)
        _cleanup(input_path, output_path)
        raise HTTPException(status_code=500, detail="Engine processing failed")

    return {
        "status": "success",
        "output_url": f"/outputs/{output_path.name}",
    }
