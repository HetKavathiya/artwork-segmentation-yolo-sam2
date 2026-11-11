# Artwork Segmentation & Auto-Crop: YOLO11-seg + SAM 2 (Computer Vision Pipeline)

**Automatically detect, segment, crop, straighten and background-remove paintings from photographs, using YOLO11 instance segmentation, a fine-tuned Segment Anything Model 2 (SAM 2.1), and classical geometry (perspective correction, ellipse fitting, edge snapping).**

Photos of paintings are hard: they are tilted, cluttered, framed, matted, round, torn-edged, or the same colour as the wall. This project turns them into clean, front-facing, transparent-background artwork images, with a review flag whenever the result is uncertain.

> **Looking for a computer vision freelancer?** I build custom image segmentation, object detection and document/artwork extraction pipelines. Jump to [Hire me](#hire-me--[freelance-computer-vision--llm-development](https://www.upwork.com/freelancers/~012293e09d67cf37c7)).

<!-- TODO: add a before/after image: ![Before and after](assets/before_after.jpg) -->

---

## What it does

- **Detects 4 classes** with a custom-trained YOLO11m-seg model: `artwork_surface`, `canvas_edge`, `frame`, `mat`.
- **Refines the masks with SAM 2.1**. YOLO provides the prompts (box, positive and negative points); SAM produces the sharper boundary.
- **Shape-aware geometry**:
  - rectangles: robust line fitting on each side, sub-pixel edge snapping, aspect-ratio estimation (Zhang–He), homography to a front-facing view;
  - circles and ovals: ellipse refit, no distortion;
  - torn or irregular edges: mask kept as-is, optional alpha matting.
- **Handles multiple paintings per photo**, each in its own folder.
- **Case-based outputs**: artwork only, artwork + mat, artwork + frame, artwork + mat + frame. The canvas edge and background are removed.
- **Transparent PNG + white-background preview** for every result.
- **Quality flags**: results are marked `success` or `needs_review` using heuristic checks. Nothing is labelled "verified" without ground truth.
- **Wide input support**: PNG, JPG, JPEG, BMP, TIFF, WEBP, HEIC, AVIF.
- **Fully local and offline** after installation.

## Pipeline

```
Photo ──► YOLO11-seg (4 classes) ──► prompts (box + points)
                                   │
                                   ▼
                         SAM 2.1 mask refinement
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                           ▼
 Rectangular artwork        Circular / oval             Irregular / torn
 line fit + edge snap       ellipse refit               keep mask (+ matting)
 + homography               (no warp)                   (no warp)
        └──────────────────────────┬───────────────────────────┘
                                   ▼
        Transparent PNG · white preview · manifest.json (flags, status)
```

## Tech stack

Python · PyTorch · Ultralytics YOLO11 (instance segmentation) · Meta SAM 2.1 · OpenCV · NumPy · Pillow (+ pillow-heif / pillow-avif) · optional PyMatting

## Quick start

```bash
git clone https://github.com/HetKavathiya/artwork-segmentation-yolo-sam2.git
cd artwork-segmentation-yolo-sam2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

python src/pipeline.py --model weights/best.pt \
    --sam2-checkpoint weights/sam2_artwork_finetuned.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
    --input path/to/images --output path/to/output
```

Model weights are not stored in this repository. See `weights/README.md`.

## Licence and attribution

- Code: <!-- TODO: choose a licence. Ultralytics YOLO is AGPL-3.0, so code that uses it is typically distributed under AGPL-3.0 unless you hold an Ultralytics enterprise licence. -->
- SAM 2 (Meta): Apache-2.0. Check the exact checkpoint terms.

---

## Hire me: freelance computer vision & LLM development

I'm a **computer vision and robotics engineer** available for freelance projects. Typical work:

- Custom **image segmentation** and **object detection** (YOLO, SAM / SAM 2, Mask R-CNN)
- **Document, artwork and product image extraction**: cropping, perspective correction, background removal
- Model **fine-tuning** and dataset pipelines (annotation, training, evaluation, error analysis)
- Production-ready **Python inference pipelines** (offline, batch, GPU/CPU)
- Robotics perception: **ROS**, LiDAR, SLAM, point clouds
- <!-- Keep only if you offer it: LLM applications (RAG, chatbots, document AI) and vision-language pipelines -->

## Contact

- Email: kavathiyahet25@gmail.com
- Upwork: https://www.upwork.com/freelancers/~012293e09d67cf37c7
- LinkedIn: https://www.linkedin.com/in/het-kavathiya-922a692aa
- Open an issue or discussion on this repository

*Keywords: computer vision freelancer, image segmentation, instance segmentation, YOLO11, SAM 2 fine-tuning, artwork detection, auto crop, perspective correction, background removal, OpenCV, PyTorch, Python, hire computer vision engineer.*
