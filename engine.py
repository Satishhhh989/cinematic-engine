
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

MAX_PROCESS_WIDTH = 640       # Downscale processing frames to this width
FRAME_SKIP = 2                # Process every Nth frame
OUTPUT_FPS_CAP = 12           # Maximum output FPS
MOTION_THRESHOLD = 25         # Binary threshold for motion mask
MIN_CONTOUR_AREA = 200        # Lower threshold = more boxes detected
MAX_MATCH_DISTANCE = 120      # Max pixel distance for centroid matching
STALE_TIMEOUT = 8             # Remove objects unseen for N processed frames
SMOOTHING_ALPHA = 0.35        # Exponential smoothing factor (lower = smoother)
TRAIL_LENGTH = 5              # Number of past centroid positions to draw
BOX_OPACITY = 0.06            # Fill opacity for bounding boxes
LABEL_FONT_SCALE = 0.35       # Font scale for floating labels
LINE_THICKNESS = 1            # Thin stroke for all drawn elements
MAX_OBJECTS = 60              # Cap tracked objects
BOX_SHRINK = 0.3              # Shrink bbox by this fraction (0.3 = 30% smaller)
MAX_BOX_SIZE = 120            # Limit bounding boxes to this max dimension to avoid huge boxes


# ═══════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """Lightweight object state for a single tracked region."""
    obj_id: int
    bbox: Tuple[int, int, int, int]           # (x, y, w, h) smoothed
    centroid: Tuple[float, float]             # (cx, cy) smoothed
    raw_bbox: Tuple[int, int, int, int]       # unsmoothed
    raw_centroid: Tuple[float, float]         # unsmoothed
    trail: List[Tuple[float, float]] = field(default_factory=list)
    prev_centroid: Optional[Tuple[float, float]] = None
    frames_since_seen: int = 0
    area: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1 — Motion Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, float], float]]:
    """Detect motion regions via frame differencing.

    Returns list of (bbox, centroid, area) tuples for valid contours.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Dilate to merge nearby motion regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        detections.append(((x, y, w, h), (cx, cy), area))

    return detections


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2 — Object Abstraction (ID Tracking)
# ═══════════════════════════════════════════════════════════════════════

class ObjectTracker:
    """Simple nearest-centroid tracker with stable object IDs."""

    def __init__(self) -> None:
        self._next_id = 1
        self._objects: Dict[int, TrackedObject] = {}

    @property
    def objects(self) -> Dict[int, TrackedObject]:
        return self._objects

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], Tuple[float, float], float]],
    ) -> Dict[int, TrackedObject]:
        """Match new detections to existing objects or create new ones."""

        # Mark all objects as unseen this frame
        for obj in self._objects.values():
            obj.frames_since_seen += 1

        # Greedy nearest-distance matching
        used_det_indices: set = set()
        used_obj_ids: set = set()

        # Build cost matrix
        obj_list = list(self._objects.values())
        if obj_list and detections:
            costs = []
            for obj in obj_list:
                row = []
                for det_bbox, det_centroid, det_area in detections:
                    dist = math.hypot(
                        obj.centroid[0] - det_centroid[0],
                        obj.centroid[1] - det_centroid[1],
                    )
                    row.append(dist)
                costs.append(row)

            # Greedy assignment by smallest distance
            flat = []
            for i, row in enumerate(costs):
                for j, d in enumerate(row):
                    flat.append((d, i, j))
            flat.sort()

            for dist, oi, di in flat:
                if oi in {idx for idx, _ in [(i, None) for i in used_obj_ids]}:
                    continue
                obj = obj_list[oi]
                if obj.obj_id in used_obj_ids or di in used_det_indices:
                    continue
                if dist > MAX_MATCH_DISTANCE:
                    continue

                det_bbox, det_centroid, det_area = detections[di]
                obj.prev_centroid = obj.centroid
                obj.raw_bbox = det_bbox
                obj.raw_centroid = det_centroid
                obj.area = det_area
                obj.frames_since_seen = 0
                used_obj_ids.add(obj.obj_id)
                used_det_indices.add(di)

        # Create new objects for unmatched detections
        for di, (det_bbox, det_centroid, det_area) in enumerate(detections):
            if di in used_det_indices:
                continue
            if len(self._objects) >= MAX_OBJECTS:
                break
            new_obj = TrackedObject(
                obj_id=self._next_id,
                bbox=det_bbox,
                centroid=det_centroid,
                raw_bbox=det_bbox,
                raw_centroid=det_centroid,
                area=det_area,
            )
            self._objects[self._next_id] = new_obj
            self._next_id += 1

        # Remove stale objects
        stale_ids = [
            oid for oid, obj in self._objects.items()
            if obj.frames_since_seen > STALE_TIMEOUT
        ]
        for oid in stale_ids:
            del self._objects[oid]

        return self._objects


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3 — Motion Smoothing
# ═══════════════════════════════════════════════════════════════════════

def smooth_objects(objects: Dict[int, TrackedObject]) -> None:
    """Apply exponential smoothing to bbox and centroid in-place."""
    a = SMOOTHING_ALPHA
    for obj in objects.values():
        if obj.frames_since_seen > 0:
            continue  # Don't smooth objects that weren't detected this frame

        # Smooth centroid
        sx = a * obj.centroid[0] + (1 - a) * obj.raw_centroid[0]
        sy = a * obj.centroid[1] + (1 - a) * obj.raw_centroid[1]
        obj.centroid = (sx, sy)

        # Smooth bounding box
        ox, oy, ow, oh = obj.bbox
        rx, ry, rw, rh = obj.raw_bbox
        obj.bbox = (
            int(a * ox + (1 - a) * rx),
            int(a * oy + (1 - a) * ry),
            int(a * ow + (1 - a) * rw),
            int(a * oh + (1 - a) * rh),
        )

        # Update trail
        obj.trail.append(obj.centroid)
        if len(obj.trail) > TRAIL_LENGTH:
            obj.trail = obj.trail[-TRAIL_LENGTH:]


# ═══════════════════════════════════════════════════════════════════════
#  Phase 4 — HUD Rendering
# ═══════════════════════════════════════════════════════════════════════

def _draw_rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    radius: int = 12,
) -> None:
    """Draw a rounded rectangle outline."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


def _draw_rounded_rect_filled(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    opacity: float,
    radius: int = 12,
) -> None:
    """Draw a filled rounded rectangle with transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(overlay, pt1, pt2, color, -1)
    else:
        # Fill corners with circles and body with rectangles
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)


def _shrink_bbox(x, y, w, h, factor):
    """Shrink a bbox inward by `factor` (0.3 = 30% smaller) and clamp to max size."""
    # Enforce maximum box size while keeping centered
    if w > MAX_BOX_SIZE:
        cx = x + w / 2.0
        w = MAX_BOX_SIZE
        x = int(cx - w / 2.0)
    if h > MAX_BOX_SIZE:
        cy = y + h / 2.0
        h = MAX_BOX_SIZE
        y = int(cy - h / 2.0)

    dx = int(w * factor * 0.5)
    dy = int(h * factor * 0.5)
    return x + dx, y + dy, max(w - 2 * dx, 4), max(h - 2 * dy, 4)


def render_hud(
    frame: np.ndarray,
    objects: Dict[int, TrackedObject],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Draw the full HUD overlay onto the frame.

    scale_x, scale_y map processing coords back to original resolution.
    """
    canvas = frame.copy()
    hud_color = (255, 255, 255)      # White
    trail_color = (180, 200, 220)    # Soft blue-grey
    link_color = (200, 220, 255)     # Light blue

    active_objects = [
        obj for obj in objects.values()
        if obj.frames_since_seen <= 2
    ]

    # ── Transparent box fills ────────────────────────────────────────
    for obj in active_objects:
        x, y, w, h = _shrink_bbox(*obj.bbox, BOX_SHRINK)
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        _draw_rounded_rect_filled(canvas, (x1, y1), (x2, y2), hud_color, BOX_OPACITY)

    # ── Motion trails ────────────────────────────────────────────────
    for obj in active_objects:
        if len(obj.trail) < 2:
            continue
        for i in range(1, len(obj.trail)):
            alpha_factor = i / len(obj.trail)
            color_faded = tuple(int(c * alpha_factor) for c in trail_color)
            pt1 = (int(obj.trail[i - 1][0] * scale_x), int(obj.trail[i - 1][1] * scale_y))
            pt2 = (int(obj.trail[i][0] * scale_x), int(obj.trail[i][1] * scale_y))
            cv2.line(canvas, pt1, pt2, color_faded, LINE_THICKNESS, cv2.LINE_AA)

    # ── Motion connection lines (prev → current centroid) ────────────
    for obj in active_objects:
        if obj.prev_centroid is not None:
            pt1 = (int(obj.prev_centroid[0] * scale_x), int(obj.prev_centroid[1] * scale_y))
            pt2 = (int(obj.centroid[0] * scale_x), int(obj.centroid[1] * scale_y))
            cv2.line(canvas, pt1, pt2, hud_color, LINE_THICKNESS, cv2.LINE_AA)

    # ── Rounded bounding boxes (shrunk) ──────────────────────────────
    for obj in active_objects:
        x, y, w, h = _shrink_bbox(*obj.bbox, BOX_SHRINK)
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        _draw_rounded_rect(canvas, (x1, y1), (x2, y2), hud_color, LINE_THICKNESS, radius=6)

    # ── Floating positional labels (x, y) ────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    for obj in active_objects:
        x, y, w, h = _shrink_bbox(*obj.bbox, BOX_SHRINK)
        lx = int(x * scale_x) + 2
        ly = int(y * scale_y) - 6
        if ly < 12:
            ly = int((y + h) * scale_y) + 12
        cx_pos = int(obj.centroid[0] * scale_x)
        cy_pos = int(obj.centroid[1] * scale_y)
        label = f"{cx_pos},{cy_pos}"
        cv2.putText(canvas, label, (lx, ly), font, LABEL_FONT_SCALE,
                    hud_color, 1, cv2.LINE_AA)

    # ── Global linking line between two largest objects ───────────────
    if len(active_objects) >= 2:
        sorted_by_area = sorted(active_objects, key=lambda o: o.area, reverse=True)
        a, b = sorted_by_area[0], sorted_by_area[1]
        pt_a = (int(a.centroid[0] * scale_x), int(a.centroid[1] * scale_y))
        pt_b = (int(b.centroid[0] * scale_x), int(b.centroid[1] * scale_y))
        cv2.line(canvas, pt_a, pt_b, link_color, LINE_THICKNESS, cv2.LINE_AA)

    return canvas


def _draw_dotted_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    gap: int = 8,
) -> None:
    """Draw a dotted/dashed line between two points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    steps = int(dist / gap)
    for i in range(0, steps, 2):
        t0 = i / steps
        t1 = min((i + 1) / steps, 1.0)
        s = (int(pt1[0] + dx * t0), int(pt1[1] + dy * t0))
        e = (int(pt1[0] + dx * t1), int(pt1[1] + dy * t1))
        cv2.line(img, s, e, color, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  Main Processing Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_engine(input_path: str, output_path: str) -> str:
    """Run the Cinematic Motion HUD Engine on a video.

    Parameters
    ----------
    input_path : str
        Path to the source video file.
    output_path : str
        Desired path for the rendered output video.

    Returns
    -------
    str
        Path to the written output file.
    """
    from video.writer import VideoWriter

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Output FPS (capped) ──────────────────────────────────────────
    out_fps = min(src_fps, OUTPUT_FPS_CAP)

    # ── Processing downscale ─────────────────────────────────────────
    if src_w > MAX_PROCESS_WIDTH:
        proc_scale = MAX_PROCESS_WIDTH / src_w
    else:
        proc_scale = 1.0
    proc_w = int(src_w * proc_scale)
    proc_h = int(src_h * proc_scale)

    # Scale factors: processing coords → original resolution
    scale_x = src_w / proc_w
    scale_y = src_h / proc_h

    log.info(
        "Source: %dx%d @ %.1f FPS (%d frames) → process at %dx%d, output at %.0f FPS",
        src_w, src_h, src_fps, total_frames, proc_w, proc_h, out_fps,
    )

    writer = VideoWriter(output_path, out_fps, (src_w, src_h))
    tracker = ObjectTracker()

    prev_gray = None
    frame_idx = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Frame skip ───────────────────────────────────────────────
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # ── Downscale for processing ─────────────────────────────────
        if proc_scale < 1.0:
            small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Phase 1 — Motion Extraction
            detections = extract_motion(prev_gray, gray)

            # Phase 2 — Object Abstraction
            objects = tracker.update(detections)

            # Phase 3 — Motion Smoothing
            smooth_objects(objects)
        else:
            objects = tracker.objects

        # Phase 4 — HUD Rendering (on original resolution frame)
        output_frame = render_hud(frame, objects, scale_x, scale_y)
        writer.write(output_frame)
        written += 1

        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()

    log.info("Done — %d frames written → %s", written, output_path)
    return output_path
