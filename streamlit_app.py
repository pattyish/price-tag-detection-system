"""Streamlit dashboard for end-to-end price tag detection and OCR review."""

from __future__ import annotations

import tempfile
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.inference.pipeline import PipelineBuilder
from src.utils.logger import PriceDetectionLogger


st.set_page_config(
    page_title="Price Tag Detection Dashboard",
    page_icon="💵",
    layout="wide",
)


NAVY = (16, 34, 61)
WHITE = (255, 255, 255)
SLATE = (98, 115, 138)
LIGHT_BG = (243, 246, 251)
SUCCESS = (32, 153, 87)
ALERT = (214, 63, 53)
INFO = (59, 130, 246)
WARNING = (211, 142, 39)
PANEL_GAP = 18


def _inject_styles() -> None:
    """Inject custom CSS for report-like Streamlit layout."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #eef3f9 0%, #f8fbff 100%);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f2747 0%, #17365d 100%);
            color: white;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 18px 40px rgba(15, 39, 71, 0.18);
            margin-bottom: 1rem;
        }
        .hero-card h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }
        .hero-card p {
            margin: 0.5rem 0 0 0;
            color: #d6e3f5;
        }
        .panel-shell {
            background: white;
            border-radius: 18px;
            border: 1px solid #d8e2ef;
            box-shadow: 0 10px 24px rgba(15, 39, 71, 0.08);
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .panel-title {
            background: #10233f;
            color: white;
            padding: 0.75rem 1rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .panel-body {
            padding: 1rem;
        }
        .kpi-card {
            background: white;
            border: 1px solid #d7e1ec;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 20px rgba(15, 39, 71, 0.06);
        }
        .kpi-label {
            color: #5f6f85;
            font-size: 0.84rem;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            color: #10233f;
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0;
        }
        .alert-box {
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
            border: 1px solid;
        }
        .alert-danger {
            background: #fff3f2;
            border-color: #f2c4c0;
            color: #9f2f26;
        }
        .alert-success {
            background: #eefaf2;
            border-color: #b8e0c5;
            color: #1f7f4c;
        }
        .small-note {
            color: #697b92;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_logging() -> None:
    """Initialize logging once for dashboard runtime."""
    if "_logging_initialized" not in st.session_state:
        PriceDetectionLogger.setup_logging(log_level="INFO")
        st.session_state["_logging_initialized"] = True


@st.cache_resource
def _load_pipeline(
    use_config: bool,
    config_path: str,
    model_name: str,
    ocr_engine: str,
    device: str,
    confidence_threshold: float,
):
    """Cache and return inference pipeline instance."""
    if use_config:
        return PipelineBuilder.create_from_config_file(config_path)
    return PipelineBuilder.create_custom(
        model_name=model_name,
        ocr_engine=ocr_engine,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def _decode_upload(uploaded_file) -> np.ndarray:
    """Decode uploaded image bytes to OpenCV BGR image."""
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image")
    return image


def _safe_crop(image: np.ndarray, box: Dict) -> Optional[np.ndarray]:
    """Return a safe crop from detection box dictionary."""
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(box.get("x_min", 0))))
    y1 = max(0, min(h - 1, int(box.get("y_min", 0))))
    x2 = max(0, min(w, int(box.get("x_max", 0))))
    y2 = max(0, min(h, int(box.get("y_max", 0))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _draw_overlay(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw detection and OCR labels on image."""
    out = image.copy()
    for det in detections:
        box = det.get("detection")
        price = det.get("price") or {}
        if not box:
            continue

        is_valid = bool(price.get("is_valid", False))
        color = (0, 220, 0) if is_valid else (0, 0, 255)
        x1, y1 = int(box["x_min"]), int(box["y_min"])
        x2, y2 = int(box["x_max"]), int(box["y_max"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        conf = float(box.get("confidence", 0.0))
        txt = price.get("formatted_price") if price.get("formatted_price") and price.get("formatted_price") != "Invalid" else (det.get("ocr_result") or {}).get("text", "")
        label = f"{txt} ({conf:.2f})" if txt else f"det ({conf:.2f})"

        label_y = y1 - 8 if y1 > 20 else y1 + 16
        cv2.putText(
            out,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _resize_to_fit(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to fit within max size preserving aspect ratio."""
    max_w, max_h = max_size
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale >= 1:
        return image.copy()
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return resized


def _place_image(canvas: np.ndarray, image: np.ndarray, area: Tuple[int, int, int, int], fill: Tuple[int, int, int] = WHITE) -> None:
    """Place an image centered inside a given area on the canvas."""
    x, y, w, h = area
    canvas[y:y + h, x:x + w] = fill
    fitted = _resize_to_fit(image, (w, h))
    fh, fw = fitted.shape[:2]
    ox = x + (w - fw) // 2
    oy = y + (h - fh) // 2
    canvas[oy:oy + fh, ox:ox + fw] = fitted


def _find_font(size: int, bold: bool = False) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """Load a reasonable font cross-platform, falling back to default."""
    candidates = []
    if bold:
        candidates.extend([
            "arialbd.ttf",
            "DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ])
    else:
        candidates.extend([
            "arial.ttf",
            "DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """Wrap text into lines constrained by max width."""
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        proposal = f"{current} {word}"
        bbox = draw.textbbox((0, 0), proposal, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = proposal
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _build_flagged_rows(rows: List[Dict], low_conf_threshold: float) -> List[Dict]:
    """Collect rejected or low-confidence rows."""
    flagged: List[Dict] = []
    for row in rows:
        is_rejected = row["Validation"] != "Valid"
        low_conf = float(row["OCR Confidence"]) < low_conf_threshold
        if is_rejected or low_conf:
            flagged.append(row)
    return flagged


def _collect_crop_items(image_bgr: np.ndarray, detections: List[Dict], limit: int = 6) -> List[Tuple[np.ndarray, str]]:
    """Collect crop images with captions."""
    items: List[Tuple[np.ndarray, str]] = []
    for idx, det in enumerate(detections):
        crop = _safe_crop(image_bgr, det.get("detection") or {})
        if crop is None:
            continue
        price = det.get("price") or {}
        ocr = det.get("ocr_result") or {}
        caption = price.get("formatted_price")
        if not caption or caption == "Invalid":
            caption = ocr.get("text") or f"Tag #{idx + 1}"
        items.append((crop, str(caption)))
        if len(items) >= limit:
            break
    return items


def _render_report_image(
    raw_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    detections: List[Dict],
    rows: List[Dict],
    flagged: List[Dict],
    model_name: str,
    ocr_engine: str,
    inference_s: float,
) -> np.ndarray:
    """Render a single combined dashboard image matching the 7-panel report style."""
    width, height = 1800, 1180
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    canvas[:] = LIGHT_BG

    title_h = 64
    cv2.rectangle(canvas, (0, 0), (width, title_h), NAVY, -1)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_title = _find_font(28, bold=True)
    font_panel = _find_font(24, bold=True)
    font_text = _find_font(20)
    font_small = _find_font(17)
    font_tiny = _find_font(15)

    draw.text((24, 16), "Retail Price Tag Detection Report", fill=WHITE, font=font_title)

    left_margin, top = 22, title_h + 18
    col_a = 610
    col_b = 610
    col_c = width - left_margin * 2 - col_a - col_b - PANEL_GAP * 2
    row1_h = 305
    row2_h = 300
    row3_h = 425

    def panel_rect(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline=(214, 223, 234), width=2)
        draw.rounded_rectangle((x, y, x + w, y + 44), radius=18, fill=NAVY)
        draw.rectangle((x, y + 22, x + w, y + 44), fill=NAVY)
        return (x, y, w, h)

    p1 = panel_rect(left_margin, top, col_a, row1_h)
    p2 = panel_rect(left_margin + col_a + PANEL_GAP, top, col_b, row1_h)
    p3 = panel_rect(left_margin + col_a + col_b + PANEL_GAP * 2, top, col_c, row1_h + row2_h + PANEL_GAP)
    p4 = panel_rect(left_margin, top + row1_h + PANEL_GAP, col_a + col_b + PANEL_GAP, row2_h)
    p5 = panel_rect(left_margin + col_a + col_b + PANEL_GAP * 2, top + row1_h + row2_h + PANEL_GAP * 2, col_c, row3_h)
    p6 = panel_rect(left_margin, top + row1_h + row2_h + PANEL_GAP * 2, col_a + col_b - 160, row3_h)
    p7 = panel_rect(left_margin + col_a + col_b - 160 + PANEL_GAP, top + row1_h + row2_h + PANEL_GAP * 2, col_c + 160 - PANEL_GAP, row3_h)

    titles = [
        (p1, "1. Raw Shelf Photo"),
        (p2, "2. Detection Overlay Result (YOLO)"),
        (p3, "3. OCR Crop Examples"),
        (p4, "4. Final Extraction Table"),
        (p5, "5. Failure / Edge-Case Example"),
        (p6, "6. Before vs After Comparison"),
        (p7, "7. Extracted Prices & Alerts Summary"),
    ]
    for rect, text in titles:
        draw.text((rect[0] + 16, rect[1] + 10), text, fill=WHITE, font=font_panel)

    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    _place_image(canvas, raw_bgr, (p1[0] + 14, p1[1] + 58, p1[2] - 28, p1[3] - 72))
    _place_image(canvas, overlay_bgr, (p2[0] + 14, p2[1] + 58, p2[2] - 28, p2[3] - 72))

    crop_items = _collect_crop_items(raw_bgr, detections, limit=6)
    crop_area_x, crop_area_y = p3[0] + 14, p3[1] + 58
    crop_area_w, crop_area_h = p3[2] - 28, p3[3] - 72
    cols = 2
    rows_grid = 3
    gap = 12
    cell_w = (crop_area_w - gap * (cols - 1)) // cols
    cell_h = (crop_area_h - gap * (rows_grid - 1)) // rows_grid
    for idx in range(rows_grid * cols):
        cx = crop_area_x + (idx % cols) * (cell_w + gap)
        cy = crop_area_y + (idx // cols) * (cell_h + gap)
        cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), (229, 235, 242), -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), (202, 213, 225), 1)
        if idx < len(crop_items):
            crop, caption = crop_items[idx]
            _place_image(canvas, crop, (cx + 8, cy + 8, cell_w - 16, cell_h - 48))
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            lines = _wrap_text(draw, caption, font_small, cell_w - 18)
            draw.text((cx + 10, cy + cell_h - 32), lines[0], fill=(44, 58, 80), font=font_small)
            canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    table_x, table_y = p4[0] + 12, p4[1] + 58
    table_w, table_h = p4[2] - 24, p4[3] - 70
    col_widths = [70, 260, 190, 150, 180, 220]
    headers = ["#", "Extracted Text", "Parsed Price", "OCR Conf", "Det Conf", "Validation"]
    scale = table_w / sum(col_widths)
    col_widths = [int(w * scale) for w in col_widths]
    row_h = max(26, min(38, table_h // max(2, len(rows) + 1)))
    x = table_x
    for i, header in enumerate(headers):
        draw.rectangle((x, table_y, x + col_widths[i], table_y + row_h), fill=NAVY)
        draw.text((x + 8, table_y + 6), header, fill=WHITE, font=font_small)
        x += col_widths[i]
    for row_idx, row in enumerate(rows[:7], start=1):
        y = table_y + row_idx * row_h
        bg = WHITE if row_idx % 2 else (246, 248, 252)
        x = table_x
        values = [
            str(row["Tag #"]),
            str(row["Extracted Text"]),
            str(row["Parsed Price"]),
            f"{float(row['OCR Confidence']):.2f}",
            f"{float(row['Detection Confidence']):.2f}",
            str(row["Validation"]),
        ]
        for i, value in enumerate(values):
            draw.rectangle((x, y, x + col_widths[i], y + row_h), fill=bg, outline=(220, 227, 236))
            fill = SUCCESS if i == 5 and value == "Valid" else (43, 57, 80)
            draw.text((x + 8, y + 6), value[:24], fill=fill, font=font_small)
            x += col_widths[i]

    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    failure_crop = None
    if flagged:
        flagged_tag = int(flagged[0]["Tag #"]) - 1
        if 0 <= flagged_tag < len(detections):
            failure_crop = _safe_crop(raw_bgr, detections[flagged_tag].get("detection") or {})
    if failure_crop is None and detections:
        failure_crop = _safe_crop(raw_bgr, detections[0].get("detection") or {})
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if failure_crop is not None:
        _place_image(canvas, failure_crop, (p5[0] + 14, p5[1] + 58, p5[2] - 28, 190))
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    info_y = p5[1] + 264
    summary_lines = []
    if flagged:
        item = flagged[0]
        summary_lines = [
            f"OCR Output: {item['Extracted Text'] or 'N/A'}",
            f"Parsed Price: {item['Parsed Price']}",
            f"Confidence: {float(item['OCR Confidence']):.2f}",
            f"Validation: {item['Validation']}",
            f"Reason: {item['Reason'] or 'Low confidence'}",
        ]
    else:
        summary_lines = [
            "No severe edge case in current image.",
            "All detected tags passed validation.",
            "Review still recommended for glare or blur.",
        ]
    for idx, line in enumerate(summary_lines):
        fill = ALERT if any(key in line for key in ["Rejected", "Reason", "Confidence"]) and flagged else (44, 58, 80)
        draw.text((p5[0] + 18, info_y + idx * 28), line, fill=fill, font=font_small)

    before_area = (p6[0] + 14, p6[1] + 58, (p6[2] - 42) // 2, p6[3] - 96)
    after_area = (p6[0] + 28 + before_area[2], p6[1] + 58, (p6[2] - 42) // 2, p6[3] - 96)
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    _place_image(canvas, raw_bgr, before_area)
    _place_image(canvas, overlay_bgr, after_area)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.rounded_rectangle((before_area[0] + 12, before_area[1] - 8, before_area[0] + 190, before_area[1] + 20), radius=10, fill=ALERT)
    draw.rounded_rectangle((after_area[0] + 12, after_area[1] - 8, after_area[0] + 210, after_area[1] + 20), radius=10, fill=SUCCESS)
    draw.text((before_area[0] + 24, before_area[1] - 3), "Before (Raw Shelf)", fill=WHITE, font=font_tiny)
    draw.text((after_area[0] + 24, after_area[1] - 3), "After (Processed Output)", fill=WHITE, font=font_tiny)

    total = len(detections)
    valid = sum(1 for row in rows if row["Validation"] == "Valid")
    flagged_count = len(flagged)
    success_rate = (valid / total) if total else 0.0
    avg_ocr = float(np.mean([float(row["OCR Confidence"]) for row in rows])) if rows else 0.0
    summary_x = p7[0] + 16
    summary_y = p7[1] + 60
    card_h = 70
    card_w = p7[2] - 32
    cards = [
        (SUCCESS, f"Total Tags Detected: {total}"),
        (SUCCESS, f"Valid Prices Extracted: {valid}"),
        (WARNING if flagged_count else SUCCESS, f"Flagged / Review: {flagged_count}"),
        (INFO, f"Success Rate: {success_rate:.1%}"),
    ]
    for idx, (color, text) in enumerate(cards):
        y = summary_y + idx * (card_h + 12)
        draw.rounded_rectangle((summary_x, y, summary_x + card_w, y + card_h), radius=14, fill=(250, 252, 255), outline=(213, 223, 234), width=2)
        draw.ellipse((summary_x + 14, y + 18, summary_x + 46, y + 50), fill=color)
        draw.text((summary_x + 60, y + 20), text, fill=(39, 53, 76), font=font_small)

    alerts_y = summary_y + 4 * (card_h + 12) + 10
    draw.text((summary_x, alerts_y), "Alerts", fill=ALERT, font=font_panel)
    if flagged:
        for idx, item in enumerate(flagged[:3]):
            box_y = alerts_y + 36 + idx * 70
            draw.rounded_rectangle((summary_x, box_y, summary_x + card_w, box_y + 58), radius=12, fill=(255, 244, 243), outline=(244, 203, 199), width=2)
            alert_text = f"Tag #{item['Tag #']} {item['Parsed Price']} - {item['Reason'] or 'Low confidence'}"
            lines = _wrap_text(draw, alert_text, font_tiny, card_w - 24)
            for line_idx, line in enumerate(lines[:2]):
                draw.text((summary_x + 12, box_y + 10 + line_idx * 18), line, fill=(151, 46, 37), font=font_tiny)
    else:
        box_y = alerts_y + 36
        draw.rounded_rectangle((summary_x, box_y, summary_x + card_w, box_y + 58), radius=12, fill=(238, 249, 242), outline=(183, 223, 198), width=2)
        draw.text((summary_x + 12, box_y + 18), "No critical alerts detected.", fill=(27, 110, 63), font=font_small)

    perf_y = p7[1] + p7[3] - 116
    draw.rounded_rectangle((summary_x, perf_y, summary_x + card_w, perf_y + 88), radius=12, fill=(241, 247, 255), outline=(194, 213, 238), width=2)
    draw.text((summary_x + 12, perf_y + 10), "System Performance", fill=NAVY, font=font_small)
    perf_text = [
        f"Detection Model: {model_name}",
        f"OCR Engine: {ocr_engine}",
        f"Average OCR Confidence: {avg_ocr:.2f}",
        f"Processing Time / Image: {inference_s:.2f}s",
    ]
    for idx, line in enumerate(perf_text):
        draw.text((summary_x + 12, perf_y + 34 + idx * 14), line, fill=(53, 69, 95), font=font_tiny)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _build_table_rows(result: Dict) -> List[Dict]:
    """Build table rows for display from result dictionary."""
    rows: List[Dict] = []
    for idx, det in enumerate(result.get("detections", []), start=1):
        box = det.get("detection") or {}
        ocr = det.get("ocr_result") or {}
        price = det.get("price") or {}
        val = det.get("validation") or {}

        rows.append(
            {
                "Tag #": idx,
                "Extracted Text": ocr.get("text", ""),
                "Parsed Price": price.get("formatted_price", "Invalid"),
                "OCR Confidence": round(float(ocr.get("confidence", 0.0)), 3),
                "Detection Confidence": round(float(box.get("confidence", 0.0)), 3),
                "Validation": "Valid" if price.get("is_valid") else "Rejected",
                "Reason": " | ".join(val.get("errors", [])) if val.get("errors") else "",
            }
        )
    return rows


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR to RGB for Streamlit display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    """Render Streamlit dashboard."""
    _init_logging()
    _inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1>Price Tag Detection Dashboard</h1>
            <p>Seven-panel report view for retail shelf auditing with YOLO detection, OCR extraction, validation, and one-click export.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Runtime Settings")
        use_config = st.checkbox("Use config file", value=True)
        config_path = st.text_input("Config path", value="config/config.yaml")

        model_name = st.selectbox("YOLO model", ["yolov8n", "yolov8s", "yolov8m", "yolov8l"], index=2)
        ocr_engine = st.selectbox("OCR engine", ["easyocr", "tesseract"], index=0)
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        confidence_threshold = st.slider("Detection confidence", 0.1, 0.95, 0.5, 0.05)
        expected_price_input = st.text_input("Expected price (optional)", value="")
        low_conf_threshold = st.slider("Flag OCR below confidence", 0.1, 0.95, 0.7, 0.05)

    uploaded_file = st.file_uploader(
        "Upload a shelf image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload one image to run the dashboard analysis.")
        return

    pipeline = _load_pipeline(
        use_config=use_config,
        config_path=config_path,
        model_name=model_name,
        ocr_engine=ocr_engine,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    run_button = st.button("Run Detection", type="primary")
    if not run_button:
        return

    expected_price = None
    if expected_price_input.strip():
        try:
            expected_price = float(expected_price_input.strip())
        except ValueError:
            st.error("Expected price must be numeric, e.g. 4.90")
            return

    with st.spinner("Running detection + OCR pipeline..."):
        try:
            uploaded_file.seek(0)
            image_bgr = _decode_upload(uploaded_file)

            suffix = Path(uploaded_file.name).suffix if Path(uploaded_file.name).suffix else ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), image_bgr)

            t0 = time.perf_counter()
            result = pipeline.process_image(str(tmp_path), expected_price=expected_price)
            inference_s = time.perf_counter() - t0

            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            st.exception(exc)
            return

    detections = result.get("detections", [])
    overlay_bgr = _draw_overlay(image_bgr, detections)

    rows = _build_table_rows(result)
    table_df = pd.DataFrame(rows)
    flagged = _build_flagged_rows(rows, low_conf_threshold)
    total = int(result.get("summary", {}).get("total_detections", 0))
    valid = int(result.get("summary", {}).get("valid_extractions", 0))
    flagged_count = max(total - valid, 0)
    success_rate = (valid / total) if total > 0 else 0.0
    avg_ocr_conf = float(np.mean([float(x["OCR Confidence"]) for x in rows])) if rows else 0.0

    report_bgr = _render_report_image(
        raw_bgr=image_bgr,
        overlay_bgr=overlay_bgr,
        detections=detections,
        rows=rows,
        flagged=flagged,
        model_name=model_name,
        ocr_engine=ocr_engine,
        inference_s=inference_s,
    )
    report_rgb = _to_rgb(report_bgr)

    st.markdown('<div class="panel-shell"><div class="panel-title">Polished Report View</div><div class="panel-body">', unsafe_allow_html=True)
    st.image(report_rgb, caption="Generated seven-panel dashboard image", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in [
        (m1, "Total Tags", str(total)),
        (m2, "Valid Prices", str(valid)),
        (m3, "Flagged", str(len(flagged))),
        (m4, "Success Rate", f"{success_rate:.1%}"),
    ]:
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown('<div class="panel-shell"><div class="panel-title">Detailed Extraction Table</div><div class="panel-body">', unsafe_allow_html=True)
        if table_df.empty:
            st.warning("No detections found in this image.")
        else:
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-shell"><div class="panel-title">Alerts and Review Queue</div><div class="panel-body">', unsafe_allow_html=True)
        if not flagged:
            st.markdown('<div class="alert-box alert-success">No edge cases were flagged for manual review.</div>', unsafe_allow_html=True)
        else:
            for item in flagged:
                st.markdown(
                    (
                        '<div class="alert-box alert-danger">'
                        f"<strong>Tag #{item['Tag #']}</strong><br/>"
                        f"OCR: {item['Extracted Text'] or 'N/A'}<br/>"
                        f"Parsed: {item['Parsed Price']}<br/>"
                        f"Confidence: {float(item['OCR Confidence']):.3f}<br/>"
                        f"Reason: {item['Reason'] or 'Low confidence'}"
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div class="small-note">Model: {model_name} | OCR: {ocr_engine} | Avg OCR confidence: {avg_ocr_conf:.3f} | Inference time: {inference_s:.3f}s</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div></div>', unsafe_allow_html=True)

    report_success, encoded = cv2.imencode('.png', report_bgr)
    report_bytes = encoded.tobytes() if report_success else b''

    st.download_button(
        label="Download JSON Result",
        data=json.dumps(result, indent=2, default=str),
        file_name="price_detection_result.json",
        mime="application/json",
    )
    if report_bytes:
        st.download_button(
            label="Download Combined Dashboard Image",
            data=report_bytes,
            file_name="price_detection_dashboard.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
