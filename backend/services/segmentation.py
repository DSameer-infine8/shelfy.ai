import math
import logging
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)

# Constants ported from notebook
TARGET_LONG_SIDE = 1800
MAX_LONG_SIDE = 2600
PADDING_CANVAS = 45
MIN_COMPONENT_AREA_RATIO = 0.01


def load_yolo_model(model_path: Union[str, Path]):
    """Load Ultralytics YOLO26-seg model for spine segmentation."""
    if YOLO is None:
        raise ImportError(
            "Ultralytics is not installed. Please install it using `pip install ultralytics>=8.4.166`."
        )
    path_str = str(model_path)
    logger.info(f"Loading YOLO segmentation model from: {path_str}")
    model = YOLO(path_str, task="segment")
    return model


def largest_foreground_contour(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Reconstruct the segmented foreground from the black background.
    Returns (largest_contour, cleaned_mask).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Black background is approximately zero in the masked crops.
    mask = (gray > 8).astype(np.uint8) * 255

    # Close gaps in the segmented spine boundary.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove tiny speckles.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return None, mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    if largest_area < MIN_COMPONENT_AREA_RATIO * image_area:
        return None, mask

    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_idx] = 255

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, cleaned

    contour = max(contours, key=cv2.contourArea)
    return contour, cleaned


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """
    Orders four points as:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.asarray(points, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def rectify_with_oriented_rectangle(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Rectify the foreground spine using its oriented minimum-area rectangle.

    Returns:
        rectified image,
        foreground mask,
        box points,
        angle of long axis before rectification,
        is_vertical (True if spine height > width in original image)
    """
    contour, mask = largest_foreground_contour(image_bgr)

    if contour is None:
        raise ValueError("Could not find a foreground spine component.")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)

    # Determine the two side lengths.
    edges = []
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        length = float(np.linalg.norm(p2 - p1))
        edges.append((length, p1, p2))

    # The long edge becomes the horizontal axis.
    long_edge = max(edges, key=lambda x: x[0])
    short_edge = min(edges, key=lambda x: x[0])

    long_len = long_edge[0]
    short_len = max(short_edge[0], 1.0)

    # Keep enough resolution for thin text.
    out_w = max(256, int(round(long_len)))
    out_h = max(64, int(round(short_len)))

    # Recover a robust 4-point rectangle and then force width > height.
    ordered = order_quad_points(box)

    top_left, top_right, bottom_right, bottom_left = ordered

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)

    quad_w = max(width_top, width_bottom)
    quad_h = max(height_right, height_left)

    is_vertical = quad_h > quad_w
    if is_vertical:
        # Swap interpretation so the long dimension becomes width.
        ordered = np.array(
            [bottom_left, top_left, top_right, bottom_right],
            dtype=np.float32
        )
        out_w, out_h = int(round(quad_h)), int(round(quad_w))
    else:
        out_w, out_h = int(round(quad_w)), int(round(quad_h))

    out_w = max(out_w, 256)
    out_h = max(out_h, 64)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
    rectified = cv2.warpPerspective(
        image_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Measure long-axis angle from the selected long edge.
    dx = long_edge[2][0] - long_edge[1][0]
    dy = long_edge[2][1] - long_edge[1][1]
    angle = math.degrees(math.atan2(dy, dx))

    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180

    return rectified, mask, box, angle, is_vertical


def crop_to_foreground_and_pad(rectified_bgr: np.ndarray) -> np.ndarray:
    """
    Remove black margins, create a clean white OCR canvas,
    and upscale while keeping aspect ratio.
    """
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    fg = (gray > 8).astype(np.uint8) * 255

    coords = cv2.findNonZero(fg)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        rectified_bgr = rectified_bgr[y:y+h, x:x+w]

    # White background helps the detector when the source mask has black outside areas.
    white_bg = np.full_like(rectified_bgr, 255)
    gray2 = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    fg2 = (gray2 > 8).astype(np.uint8) * 255

    clean = white_bg.copy()
    clean[fg2 > 0] = rectified_bgr[fg2 > 0]

    h, w = clean.shape[:2]
    scale = TARGET_LONG_SIDE / max(h, w)
    scale = max(scale, 1.0)

    new_w = min(MAX_LONG_SIDE, max(256, int(round(w * scale))))
    new_h = max(64, int(round(h * (new_w / max(w, 1)))))

    clean = cv2.resize(clean, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    clean = cv2.copyMakeBorder(
        clean,
        PADDING_CANVAS, PADDING_CANVAS, PADDING_CANVAS, PADDING_CANVAS,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return clean


def rectify_spine(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Full rectification pipeline:
    1. Geometric rectification
    2. Foreground crop + white padding + upscale
    3. Rotation correction:
       - Vertical book spines (height > width on shelf) are rotated 180°
         to align top-to-bottom English spine text left-to-right.
       - Horizontal book spines (width >= height, e.g. horizontal book stacks)
         are kept upright at 0°.
    """
    try:
        rectified, mask, box, angle, is_vertical = rectify_with_oriented_rectangle(image_bgr)
        clean = crop_to_foreground_and_pad(rectified)
        # Only rotate 180° if the spine was originally vertical
        if is_vertical:
            clean = cv2.rotate(clean, cv2.ROTATE_180)
        return {
            "image": clean,
            "mask": mask,
            "box": box,
            "angle": angle,
            "is_vertical": is_vertical,
            "status": "ok"
        }
    except Exception as e:
        logger.warning(f"Geometric rectification fallback triggered: {e}")
        # Fallback: make white canvas and rotate if needed
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        fg = (gray > 8).astype(np.uint8)
        white_bg = np.full_like(image_bgr, 255)
        white_bg[fg > 0] = image_bgr[fg > 0]
        h_orig, w_orig = image_bgr.shape[:2]
        fallback_vertical = h_orig > w_orig
        clean_fallback = white_bg
        if fallback_vertical:
            clean_fallback = cv2.rotate(clean_fallback, cv2.ROTATE_90_CLOCKWISE)
        return {
            "image": clean_fallback,
            "mask": fg * 255,
            "box": np.array([]),
            "angle": 0.0,
            "is_vertical": fallback_vertical,
            "status": f"fallback: {e}"
        }


SPINE_PALETTE_BGR = [
    (70, 194, 236),   # Amber / Warm Gold
    (238, 187, 34),   # Cyan / Teal
    (130, 214, 110),  # Emerald Green
    (225, 105, 180),  # Radiant Violet
    (100, 149, 237),  # Coral
    (255, 191, 0),    # Sky Blue
    (0, 215, 255),    # Bright Gold
    (180, 105, 255),  # Deep Pink
]


def create_spine_segmentation_overlay(
    image_bgr: np.ndarray,
    book_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[List[List[float]]]]:
    """
    Renders high-quality translucent segmentation polygons and numbered badges
    over the bookshelf image. Returns:
      - annotated_image: base64 JPEG data URI ("data:image/jpeg;base64,...")
      - normalized_polygons: list of normalized [x, y] coordinates (0.0 to 1.0)
    """
    if not book_candidates or image_bgr is None:
        return None, []

    try:
        img_h, img_w = image_bgr.shape[:2]
        if img_h == 0 or img_w == 0:
            return None, []

        overlay = image_bgr.copy()
        mask_layer = np.zeros_like(image_bgr)
        normalized_polygons = []

        # 1. Fill translucent polygons
        for i, book in enumerate(book_candidates):
            poly_arr = np.asarray(book["polygon"], dtype=np.int32)
            if len(poly_arr) < 3:
                continue
            color = SPINE_PALETTE_BGR[i % len(SPINE_PALETTE_BGR)]
            cv2.fillPoly(mask_layer, [poly_arr], color)

            # Normalize polygon points to [0.0, 1.0] for responsive frontend SVG rendering
            norm_poly = [
                [round(float(pt[0]) / max(1, img_w), 4), round(float(pt[1]) / max(1, img_h), 4)]
                for pt in poly_arr
            ]
            normalized_polygons.append(norm_poly)

        # Blend translucent fill (35% polygon color, 65% original image)
        cv2.addWeighted(mask_layer, 0.35, overlay, 0.65, 0, overlay)

        # 2. Draw crisp anti-aliased borders and spine index badges
        for i, book in enumerate(book_candidates):
            poly_arr = np.asarray(book["polygon"], dtype=np.int32)
            if len(poly_arr) < 3:
                continue
            color = SPINE_PALETTE_BGR[i % len(SPINE_PALETTE_BGR)]
            cv2.polylines(overlay, [poly_arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Draw numbered badge at top-center of spine polygon
            bx, by, bw, bh = cv2.boundingRect(poly_arr)
            cx = int(bx + bw // 2)
            cy = max(16, int(by + 16))
            cx = max(14, min(img_w - 14, cx))
            cy = max(14, min(img_h - 14, cy))

            cv2.circle(overlay, (cx, cy), 13, (20, 20, 20), -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 13, color, 1, lineType=cv2.LINE_AA)
            text = str(i + 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(
                overlay,
                text,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 3. Encode to JPEG base64 data URI
        success, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            return None, normalized_polygons

        b64_str = base64.b64encode(buf).decode("utf-8")
        annotated_uri = f"data:image/jpeg;base64,{b64_str}"
        return annotated_uri, normalized_polygons
    except Exception as e:
        logger.warning(f"Error creating spine segmentation overlay: {e}")
        return None, []


def segment_and_crop_spines(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 960,
    padding: int = 5,
    return_annotated: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Optional[str], List[List[List[float]]]]]:
    """
    Run YOLO26-seg spine segmentation, sort detected book spines left-to-right,
    crop individual spines, and perform geometric rectification.

    If return_annotated is True, returns (cropped_spines, annotated_image_b64, normalized_polygons).
    Otherwise returns cropped_spines.
    """
    # Invariant: Run spine segmentation with imgsz=960, conf=0.25, retina_masks=True, verbose=False
    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        retina_masks=True,
        verbose=False
    )

    if not results or len(results) == 0:
        logger.info("No prediction results from YOLO.")
        if return_annotated:
            return [], None, []
        return []

    result = results[0]

    # Invariant: Guard against result.masks is None
    if result.masks is None or result.masks.xy is None or len(result.masks.xy) == 0:
        logger.info("No segmentation masks detected by YOLO.")
        if return_annotated:
            return [], None, []
        return []

    polygons = result.masks.xy
    logger.info(f"Detected {len(polygons)} book spine polygons.")

    img_h, img_w = image_bgr.shape[:2]

    # 1. Collect polygons & compute center_x for left-to-right sorting
    book_candidates = []
    for polygon in polygons:
        poly_arr = np.asarray(polygon, dtype=np.int32)
        if len(poly_arr) < 3:
            continue

        x_min = int(np.min(poly_arr[:, 0]))
        y_min = int(np.min(poly_arr[:, 1]))
        x_max = int(np.max(poly_arr[:, 0]))
        y_max = int(np.max(poly_arr[:, 1]))
        center_x = (x_min + x_max) / 2.0

        book_candidates.append({
            "polygon": poly_arr,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "center_x": center_x,
        })

    # Check if books are predominantly horizontally stacked (width > height)
    horizontal_count = sum(1 for b in book_candidates if (b["x_max"] - b["x_min"]) > (b["y_max"] - b["y_min"]))
    is_stack = horizontal_count > (len(book_candidates) / 2)

    if is_stack:
        # Sort top -> bottom for horizontal stacks
        book_candidates.sort(key=lambda item: (item["y_min"] + item["y_max"]) / 2.0)
    else:
        # Sort left -> right for vertical bookshelves
        book_candidates.sort(key=lambda item: item["center_x"])

    # Generate annotated image & normalized polygons if requested
    annotated_image_uri: Optional[str] = None
    normalized_polygons: List[List[List[float]]] = []
    if return_annotated:
        annotated_image_uri, normalized_polygons = create_spine_segmentation_overlay(
            image_bgr=image_bgr,
            book_candidates=book_candidates
        )

    # 2. Crop each spine using polygon mask
    cropped_spines = []
    for spine_idx, book in enumerate(book_candidates, start=1):
        polygon = book["polygon"]

        x_min = max(0, book["x_min"] - padding)
        y_min = max(0, book["y_min"] - padding)
        x_max = min(img_w, book["x_max"] + padding)
        y_max = min(img_h, book["y_max"] + padding)

        if x_max <= x_min or y_max <= y_min:
            continue

        # Create binary mask for this polygon
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(full_mask, [polygon], 255)

        crop_mask = full_mask[y_min:y_max, x_min:x_max]
        crop_bgr = image_bgr[y_min:y_max, x_min:x_max]

        # Apply mask to leave background black
        masked_crop = cv2.bitwise_and(crop_bgr, crop_bgr, mask=crop_mask)

        # Geometric rectification for OCR
        rectified_result = rectify_spine(masked_crop)
        rectified_image = rectified_result["image"]

        cropped_spines.append({
            "spine_index": spine_idx,
            "bbox": [x_min, y_min, x_max, y_max],
            "polygon": polygon.tolist(),
            "raw_crop": masked_crop,
            "rectified_image": rectified_image,
            "angle": rectified_result.get("angle", 0.0),
            "is_vertical": rectified_result.get("is_vertical", True),
        })

    logger.info(f"Successfully cropped and rectified {len(cropped_spines)} book spines.")
    if return_annotated:
        return cropped_spines, annotated_image_uri, normalized_polygons
    return cropped_spines
