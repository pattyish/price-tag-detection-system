# High-Level Architecture

```mermaid
flowchart TD
    A[Input Sources<br/>Shelf Camera / Mobile Upload / Batch Folder] --> B[Ingestion Layer<br/>Validation + Decode + Resize]
    B --> C[YOLOv8 Price Tag Detector]
    C --> D[Detected Price Tag Regions<br/>Bounding Boxes + Scores]
    D --> E[Region Cropping + Enhancement]
    E --> F[OCR Engine<br/>EasyOCR or Tesseract]
    F --> G[Post-Processing<br/>Regex + Currency + Confidence Rules]
    G --> H[Price Validation<br/>Range + Catalog Match + Anomaly Checks]
    H --> I[Structured Output<br/>JSON + API Response]
    I --> J[Downstream Systems<br/>Pricing Audit / Alerts / BI / Compliance]

    C --> K[Model Metrics]
    F --> K
    H --> K
    K --> L[Observability<br/>Logs + Counters + Error Tracking]
```

## Flow Summary

1. Input images arrive from shelf cameras, mobile uploads, or batch image folders.
2. The ingestion layer validates and normalizes the image before inference.
3. YOLOv8 detects shelf price-tag regions.
4. Detected regions are cropped and optionally enhanced for OCR.
5. OCR extracts the visible text from each detected price tag.
6. Post-processing parses prices, currencies, and confidence values.
7. Validation checks plausibility, expected values, and anomalies.
8. Structured results feed dashboards, APIs, alerts, and retail audit workflows.
