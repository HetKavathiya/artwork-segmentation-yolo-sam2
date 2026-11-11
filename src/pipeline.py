from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except ImportError:
    _HEIF_OK = False

try:
    import pillow_avif
    _AVIF_OK = True
except ImportError:
    _AVIF_OK = False


log = logging.getLogger("artwork")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8"),
        ],
    )


@dataclass
class Paths:
    model_path: str = "models/best.pt"
    input_dir: str = "input"
    output_dir: str = "output"
    sam2_checkpoint: str = ""
    sam2_config: str = ""


@dataclass
class YoloCfg:
    device: str = ""
    conf: float = 0.25
    nms_iou: float = 0.70
    imgsz: int = 1024


@dataclass
class ClassCfg:
    surface: str = "artwork_surface"
    canvas_edge: str = "canvas_edge"
    frame: str = "frame"
    mat: str = "mat"


@dataclass
class Sam2Cfg:
    enabled: bool = True
    box_expand: float = 0.03
    min_iou: float = 0.65
    n_points: int = 4
    max_grow_frac: float = 0.05


@dataclass
class RefineCfg:
    grabcut_enabled: bool = True
    grabcut_band_frac: float = 0.02
    grabcut_iters: int = 4
    grabcut_max_side: int = 1000


@dataclass
class ContentCfg:
    enabled: bool = True
    sat_otsu_blend: float = 0.5
    edge_lo: int = 30
    edge_hi: int = 100
    region_dilate_frac: float = 0.15
    min_content_ratio: float = 0.25
    max_content_ratio: float = 0.98
    min_support_gain: float = 0.05


@dataclass
class GeometryCfg:
    quad_iou_exact: float = 0.96
    quad_iou_rough: float = 0.88
    ellipse_iou_min: float = 0.85
    natural_poly_eps_frac: float = 0.008
    snap_edges: bool = True
    snap_band_frac: float = 0.006
    snap_min_grad: float = 3.0
    warp_max_px: int = 40_000_000
    edge_inset_px: float = 0.0
    feather_sigma: float = 0.8
    w_boundary: float = 0.45
    w_agreement: float = 0.30
    w_area: float = 0.15
    w_regularity: float = 0.10


@dataclass
class OutputCfg:
    jpeg_quality: int = 98
    pad_frac: float = 0.01
    exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                   ".webp", ".heic", ".heif", ".avif")


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    yolo: YoloCfg = field(default_factory=YoloCfg)
    classes: ClassCfg = field(default_factory=ClassCfg)
    sam2: Sam2Cfg = field(default_factory=Sam2Cfg)
    refine: RefineCfg = field(default_factory=RefineCfg)
    content: ContentCfg = field(default_factory=ContentCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    output: OutputCfg = field(default_factory=OutputCfg)


def load_config(path: str | None) -> Config:
    cfg = Config()
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for key, cls in [
            ("paths", Paths), ("yolo", YoloCfg), ("classes", ClassCfg),
            ("sam2", Sam2Cfg), ("refine", RefineCfg), ("content", ContentCfg),
            ("geometry", GeometryCfg), ("output", OutputCfg),
        ]:
            if key in raw:
                setattr(cfg, key, cls(**raw[key]))
        log.info("Loaded config from %s", path)
    return cfg


def read_image(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        import pillow_avif
    except ImportError:
        pass

    suffix = path.suffix.lower()
    if suffix in (".heic", ".heif") and not _HEIF_OK:
        raise RuntimeError(f"{path.name}: install pillow-heif")
    if suffix == ".avif" and not _AVIF_OK:
        raise RuntimeError(f"{path.name}: install pillow-avif-plugin")

    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("I;16", "I;16B", "I", "F"):
        a = np.asarray(im).astype(np.float32)
        im = Image.fromarray((a / max(float(a.max()), 1.0) * 255).astype(np.uint8))
    elif im.mode in ("P", "LA"):
        im = im.convert("RGBA")

    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg.convert("RGB")
    else:
        im = im.convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def write_image(path: Path, img: np.ndarray, quality: int = 98) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"encode failed: {path}")
    buf.tofile(str(path))


def unique_dir(p: Path) -> Path:
    q, i = p, 2
    while q.exists():
        q = p.with_name(f"{p.name}_run{i:02d}")
        i += 1
    q.mkdir(parents=True)
    return q


def jdefault(o: Any) -> Any:
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    i = np.count_nonzero((a > 0) & (b > 0))
    u = np.count_nonzero((a > 0) | (b > 0))
    return i / u if u else 0.0


def contour_of(mask):
    if mask is None or mask.size == 0:
        return None
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea) if cs else None


def largest_component(mask):
    if mask is None or not mask.any():
        return np.zeros_like(mask) if mask is not None else np.zeros((1, 1), np.uint8), 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask.copy(), 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    out = (lab == k).astype(np.uint8) * 255
    return out, int(np.count_nonzero(mask) - np.count_nonzero(out))


def order_quad(q):
    s, d = q.sum(1), np.diff(q, axis=1)[:, 0]
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag / (mag.max() + 1e-6)


def load_yolo(model_path: str):
    from ultralytics import YOLO
    m = YOLO(model_path)
    if getattr(m, "task", None) != "segment":
        raise ValueError(f"Model task = {m.task}, need segmentation")
    log.info("YOLO loaded. Classes: %s", m.names)
    return m


def yolo_predict(model, img, cfg: Config):
    kw = dict(imgsz=cfg.yolo.imgsz, conf=cfg.yolo.conf, iou=cfg.yolo.nms_iou,
              retina_masks=True, verbose=False)
    if cfg.yolo.device:
        kw["device"] = cfg.yolo.device
    r = model.predict(img, **kw)[0]
    H, W = img.shape[:2]
    out = []
    if r.masks is None or r.boxes is None:
        return out
    md = r.masks.data.cpu().numpy()
    for i in range(len(r.boxes)):
        m = md[i]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        c = int(r.boxes.cls[i])
        out.append(dict(cls=c, name=r.names[c], conf=float(r.boxes.conf[i]),
                        box=[float(v) for v in r.boxes.xyxy[i].tolist()],
                        mask=((m > 0.5).astype(np.uint8) * 255)))
    return out


def dedupe(dets):
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        if all(iou(d["mask"], k["mask"]) < 0.7 for k in keep):
            keep.append(d)
    return keep


def content_mask(img: np.ndarray, yolo_mask: np.ndarray,
                 cfg: ContentCfg) -> np.ndarray | None:
    if not yolo_mask.any():
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_blur = cv2.GaussianBlur(sat, (0, 0), 3)
    _, sat_bin = cv2.threshold(sat_blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.edge_lo, cfg.edge_hi)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    a = cfg.sat_otsu_blend
    score = a * (sat_bin > 0).astype(np.float32) + (1 - a) * (edges > 0).astype(np.float32)
    content = (score > 0.5).astype(np.uint8) * 255

    ys, xs = np.where(yolo_mask > 0)
    if len(xs) == 0:
        return None
    d = int(cfg.region_dilate_frac * np.sqrt(len(xs))) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, d), max(3, d)))
    region = cv2.dilate(yolo_mask, k)
    content = content & (region > 0)

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, k2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN, k3)
    content, _ = largest_component(content)

    a_cnt = np.count_nonzero(content)
    a_yolo = np.count_nonzero(yolo_mask)
    if a_cnt < 100:
        return None
    ratio = a_cnt / max(a_yolo, 1)
    if ratio < cfg.min_content_ratio or ratio > cfg.max_content_ratio:
        return None
    return content


def load_sam2(cfg: Config):
    if not (cfg.paths.sam2_checkpoint and cfg.paths.sam2_config):
        log.warning("SAM2 not configured — using grabCut")
        return None
    if not Path(cfg.paths.sam2_checkpoint).exists():
        log.warning("SAM2 ckpt not found: %s", cfg.paths.sam2_checkpoint)
        return None
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        dev = cfg.yolo.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading SAM2 on %s ...", dev)
        return SAM2ImagePredictor(
            build_sam2(cfg.paths.sam2_config, cfg.paths.sam2_checkpoint, device=dev))
    except Exception as e:
        log.warning("SAM2 load failed: %s", e)
        return None


def sample_points(mask: np.ndarray, k: int, min_dt_frac: float = 0.5) -> np.ndarray:
    if mask is None or not mask.any():
        return np.empty((0, 2), np.float32)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.where(dt >= max(min_dt_frac * dt.max(), 0.5))
    if len(xs) == 0:
        return np.empty((0, 2), np.float32)
    if len(xs) > 3000:
        s = np.random.default_rng(0).choice(len(xs), 3000, replace=False)
        xs, ys = xs[s], ys[s]
    i = int(np.argmax(dt[ys, xs]))
    pts = [(xs[i], ys[i])]
    for _ in range(k - 1):
        d = np.min([(xs - px) ** 2 + (ys - py) ** 2 for px, py in pts], axis=0)
        j = int(np.argmax(d))
        pts.append((xs[j], ys[j]))
    return np.array(pts, np.float32)


def sam2_refine(predictor, img, ref_mask, box, cfg: Config) -> np.ndarray:
    if predictor is None:
        return ref_mask
    try:
        import torch
        H, W = img.shape[:2]
        x0, y0, x1, y1 = box
        e = cfg.sam2.box_expand * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(x0 - e, 0)), int(max(y0 - e, 0))
        X1, Y1 = int(min(x1 + e, W)), int(min(y1 + e, H))
        if X1 <= X0 or Y1 <= Y0:
            return ref_mask
        roi = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2RGB)
        pos = sample_points(ref_mask[Y0:Y1, X0:X1], cfg.sam2.n_points)
        if len(pos) == 0:
            return ref_mask
        neg_mask = (ref_mask[Y0:Y1, X0:X1] == 0).astype(np.uint8) * 255
        neg = sample_points(neg_mask, 3)
        pts = np.vstack([pos, neg]) if len(neg) else pos
        lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
        b = np.array([x0 - X0, y0 - Y0, x1 - X0, y1 - Y0], np.float32)
        b[0] = max(b[0], 0); b[1] = max(b[1], 0)
        b[2] = min(b[2], X1 - X0); b[3] = min(b[3], Y1 - Y0)
        with torch.inference_mode():
            predictor.set_image(roi)
            masks, scores, _ = predictor.predict(
                point_coords=pts.astype(np.float32),
                point_labels=lab, box=b, multimask_output=True)
        best_m, best_score = None, -1
        for m, s in zip(masks, scores):
            full = np.zeros((H, W), np.uint8)
            full[Y0:Y1, X0:X1] = (m > 0).astype(np.uint8) * 255
            combined = iou(full, ref_mask) + 0.1 * float(s)
            if combined > best_score:
                best_score, best_m = combined, full
        if best_m is None:
            return ref_mask
        if iou(best_m, ref_mask) < cfg.sam2.min_iou:
            return ref_mask
        k = int(max(3, cfg.sam2.max_grow_frac * np.sqrt(np.count_nonzero(ref_mask))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        refined = best_m & cv2.dilate(ref_mask, kernel)
        refined, _ = largest_component(refined)
        return refined
    except Exception as e:
        log.debug("SAM2 refine failed: %s", e)
        return ref_mask


def grabcut_refine(img, mask, cfg: Config) -> np.ndarray:
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    bw = max(4, int(cfg.refine.grabcut_band_frac * np.sqrt(len(xs))))
    x0, x1 = max(xs.min() - 2 * bw, 0), min(xs.max() + 2 * bw + 1, W)
    y0, y1 = max(ys.min() - 2 * bw, 0), min(ys.max() + 2 * bw + 1, H)
    roi = img[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1].copy()
    scale = min(1.0, cfg.refine.grabcut_max_side / max(roi.shape[:2]))
    if scale < 1:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = max(2, int(bw * scale))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
    fg = cv2.erode(m, k)
    bg = cv2.dilate(m, k)
    gc = np.where(m > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[fg > 0] = cv2.GC_FGD
    gc[bg == 0] = cv2.GC_BGD
    try:
        cv2.grabCut(roi, gc, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    cfg.refine.grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    r = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    if scale < 1:
        r = cv2.resize(r, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = r
    out, _ = largest_component(out)
    cs, h = cv2.findContours(out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if h is not None:
        for i, c in enumerate(cs):
            if h[0][i][3] != -1 and cv2.contourArea(c) < 0.005 * np.count_nonzero(out):
                cv2.drawContours(out, [c], -1, 255, -1)
    return out


def refine_mask(img, surface, sam_predictor, box, cfg: Config) -> np.ndarray:
    if sam_predictor is not None:
        r = sam2_refine(sam_predictor, img, surface, box, cfg)
        if r.any() and iou(r, surface) >= cfg.sam2.min_iou:
            return r
    if cfg.refine.grabcut_enabled:
        try:
            r = grabcut_refine(img, surface, cfg)
            if r.any() and iou(r, surface) >= 0.5:
                return r
        except Exception:
            pass
    return surface


def boundary_support(gray: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    c = contour_of(mask)
    if c is None:
        return 0.0
    grad = gradient_magnitude(gray)
    grad_dil = cv2.dilate(grad, np.ones((5, 5), np.uint8))
    pts = c[:, 0, :]
    H, W = grad_dil.shape
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)
    return float(grad_dil[ys, xs].mean())


def hypothesis_score(gray: np.ndarray, hypothesis: np.ndarray,
                     yolo_ref: np.ndarray, regularity: float,
                     cfg: GeometryCfg) -> float:
    if hypothesis is None or not hypothesis.any():
        return -1.0
    bs = boundary_support(gray, hypothesis)
    ag = iou(hypothesis, yolo_ref)
    a_h = np.count_nonzero(hypothesis)
    a_r = max(np.count_nonzero(yolo_ref), 1)
    ratio = a_h / a_r
    area_score = float(np.exp(-2.0 * (np.log(max(ratio, 1e-6))) ** 2))
    return (cfg.w_boundary * bs + cfg.w_agreement * ag +
            cfg.w_area * area_score + cfg.w_regularity * regularity)


def robust_quad(mask: np.ndarray):
    c = contour_of(mask)
    if c is None or len(c) < 8:
        return None
    per = cv2.arcLength(c, True)
    if per <= 0:
        return None
    hull = cv2.convexHull(c)
    q0 = None
    for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
        a = cv2.approxPolyDP(hull, eps * per, True)
        if len(a) == 4:
            q0 = a[:, 0, :].astype(np.float32)
            break
    if q0 is None:
        q0 = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    q0 = order_quad(q0)
    pts = c[:, 0, :].astype(np.float32)
    tol = max(3.0, 0.02 * np.sqrt(cv2.contourArea(c)))
    lines = []
    for i in range(4):
        p, r = q0[i], q0[(i + 1) % 4]
        d = r - p
        L = np.linalg.norm(d)
        if L < 1:
            return None
        u = d / L
        n = np.array([-u[1], u[0]])
        t, dist = (pts - p) @ u, np.abs((pts - p) @ n)
        sel = pts[(dist < tol) & (t > 0.1 * L) & (t < 0.9 * L)]
        if len(sel) < 10:
            return None
        try:
            vx, vy, x0, y0 = cv2.fitLine(sel.reshape(-1, 1, 2),
                                          cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        except cv2.error:
            return None
        lines.append((np.array([x0, y0], np.float32),
                      np.array([vx, vy], np.float32)))

    def _isect(l1, l2):
        (p1, u1), (p2, u2) = l1, l2
        A = np.array([u1, -u2], np.float32).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t = np.linalg.solve(A.astype(np.float64), (p2 - p1).astype(np.float64))
        return p1 + float(t[0]) * u1

    q = [_isect(lines[i - 1], lines[i]) for i in range(4)]
    if any(v is None for v in q):
        return None
    q = np.array(q, np.float32)
    diag = np.linalg.norm(q0[0] - q0[2])
    if diag < 1 or np.max(np.linalg.norm(q - q0, axis=1)) > 0.06 * diag:
        return None
    if not cv2.isContourConvex(q.astype(np.int32)):
        return None
    return q


def ellipse_hypothesis(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    c = contour_of(mask)
    if c is None or len(c) < 20:
        return None, 0.0
    try:
        e = cv2.fitEllipse(c)
    except cv2.error:
        return None, 0.0
    em = np.zeros_like(mask)
    cv2.ellipse(em, e, 255, -1)
    return em, iou(mask, em)


def natural_polygon(mask: np.ndarray, eps_frac: float) -> np.ndarray | None:
    c = contour_of(mask)
    if c is None or len(c) < 10:
        return None
    per = cv2.arcLength(c, True)
    if per <= 0:
        return None
    poly = cv2.approxPolyDP(c, eps_frac * per, True)
    if len(poly) < 3:
        return None
    out = np.zeros_like(mask)
    cv2.fillPoly(out, [poly.astype(np.int32)], 255)
    return out


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    k = 5
    m = cv2.medianBlur(mask, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return m


def generate_hypotheses(mask: np.ndarray, gray: np.ndarray,
                        cfg: GeometryCfg) -> dict[str, tuple[np.ndarray | None, float, np.ndarray | None]]:
    out: dict = {}

    q = robust_quad(mask)
    if q is not None:
        qm = np.zeros_like(mask)
        cv2.fillPoly(qm, [q.astype(np.int32)], 255)
        qiou = iou(mask, qm)
        out["quad"] = (qm, qiou, q)

    em, eiou = ellipse_hypothesis(mask)
    if em is not None:
        out["ellipse"] = (em, eiou, None)

    npoly = natural_polygon(mask, cfg.natural_poly_eps_frac)
    if npoly is not None:
        out["natural_polygon"] = (npoly, 0.5, None)

    sm = smooth_mask(mask)
    out["smoothed"] = (sm, 0.3, None)

    out["raw"] = (mask, 0.0, None)

    return out


def snap_quad_edges(q, gray, band_frac, min_grad):
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    c = q.mean(0)
    lines, notes = [], []
    for i in range(4):
        p, r = q[i], q[(i + 1) % 4]
        d = r - p
        L = float(np.linalg.norm(d))
        u = d / max(L, 1e-6)
        n = np.array([-u[1], u[0]], np.float32)
        if np.dot(n, (p + r) / 2 - c) < 0:
            n = -n
        band = max(2.0, band_frac * L)
        ts = np.arange(-band, band + 1e-6, 0.5, dtype=np.float32)
        P = p[None, :] + np.linspace(0.1, 0.9, 80, dtype=np.float32)[:, None] * d[None, :]
        D = (gx * n[0] + gy * n[1]).astype(np.float32)
        X = (P[:, None, 0] + ts[None, :] * n[0]).astype(np.float32)
        Y = (P[:, None, 1] + ts[None, :] * n[1]).astype(np.float32)
        V = np.abs(cv2.remap(D, X, Y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE))
        tb = ts[V.argmax(1)]
        ok = (np.median(V.max(1)) >= min_grad and
              np.median(np.abs(tb - np.median(tb))) <= 1.5)
        s = float(np.median(tb)) if ok else 0.0
        notes.append(dict(side=i, shift_px=round(s, 2), applied=bool(ok)))
        lines.append((p + n * s, u))

    def _isect(l1, l2):
        (p1, u1), (p2, u2) = l1, l2
        A = np.array([u1, -u2], np.float32).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        t = np.linalg.solve(A.astype(np.float64), (p2 - p1).astype(np.float64))
        return p1 + float(t[0]) * u1

    q2 = [_isect(lines[i - 1], lines[i]) for i in range(4)]
    return (np.array(q2, np.float32) if all(v is not None for v in q2) else q), notes


def aspect_zhang_he(q, shape_hw):
    H, W = shape_hw
    m = [np.array([x - W / 2, y - H / 2, 1.0]) for x, y in q]
    m1, m2, m4, m3 = m[0], m[1], m[2], m[3]
    cr = np.cross
    try:
        k2 = (cr(m1, m4) @ m3) / (cr(m2, m4) @ m3)
        k3 = (cr(m1, m4) @ m2) / (cr(m3, m4) @ m2)
        n2, n3 = k2 * m2 - m1, k3 * m3 - m1
        den = n2[2] * n3[2]
        if abs(den) < 1e-9:
            return None
        f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / den
        if not np.isfinite(f2) or f2 <= 0:
            return None
        w = np.sqrt((n2[0] ** 2 + n2[1] ** 2) / f2 + n2[2] ** 2)
        h = np.sqrt((n3[0] ** 2 + n3[1] ** 2) / f2 + n3[2] ** 2)
        return float(w / h)
    except Exception:
        return None


def pick_aspect(q, shape_hw):
    zh = aspect_zhang_he(q, shape_hw)
    tl, tr, br, bl = q
    d = lambda a, b: float(np.linalg.norm(a - b))
    edge = max(d(tl, tr), d(bl, br)) / max(d(tl, bl), d(tr, br))
    if zh is not None and 0.2 < zh < 5 and abs(zh / edge - 1) < 0.4:
        return zh, "zhang_he"
    return edge, "edge_lengths"


def warp_quad(img, mask, q, aspect, max_px):
    tl, tr, br, bl = q
    Wd = max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br))
    Hd = Wd / max(aspect, 1e-6)
    s = min(1.0, np.sqrt(max_px / max(Wd * Hd, 1)))
    Wd, Hd = Wd * s, Hd * s
    if Wd < 8 or Hd < 8:
        return None
    W_out, H_out = int(round(Wd)), int(round(Hd))
    dst = np.float32([[0, 0], [W_out - 1, 0],
                      [W_out - 1, H_out - 1], [0, H_out - 1]])
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    wi = cv2.warpPerspective(img, M, (W_out, H_out), flags=cv2.INTER_CUBIC)
    wm = cv2.warpPerspective(mask, M, (W_out, H_out), flags=cv2.INTER_LINEAR)
    return wi, wm, dict(output_size=[W_out, H_out],
                        corners_xy=q.tolist(),
                        aspect_ratio=round(aspect, 4))


def alpha_from_mask(mask, edge_inset, feather_sigma):
    m = mask
    if edge_inset > 0:
        m = ((cv2.distanceTransform(mask, cv2.DIST_L2, 3) > edge_inset) * 255).astype(np.uint8)
    if feather_sigma > 0:
        m = np.where(m > 0, cv2.GaussianBlur(m, (0, 0), feather_sigma), 0).astype(np.uint8)
    return m


def tight_crop(rgb, alpha, thr=8):
    ys, xs = np.where(alpha > thr)
    if not len(xs):
        return rgb, alpha
    s = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return rgb[s], alpha[s]


def save_variant(out_dir: Path, name: str, rgb: np.ndarray, alpha: np.ndarray,
                 quality: int) -> list[str]:
    files = []
    if alpha is None or alpha.min() == 255:
        p = out_dir / f"{name}.png"
        write_image(p, rgb, quality)
        return [str(p)]
    rgb, alpha = tight_crop(rgb, alpha)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    p_t = out_dir / f"{name}_transparent.png"
    write_image(p_t, np.dstack([rgb, alpha]), quality)
    files.append(str(p_t))
    a = alpha[..., None].astype(np.float32) / 255
    white = (rgb.astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    p_w = out_dir / f"{name}_white_preview.jpg"
    write_image(p_w, white, quality)
    files.append(str(p_w))
    return files


def save_mask(out_dir: Path, name: str, mask: np.ndarray, quality: int) -> str:
    p = out_dir / f"mask_{name}.png"
    write_image(p, mask, quality)
    return str(p)


def render_variant(img, gray, vmask, name, out_dir,
                   cfg: Config, flags, info) -> list[str]:
    if not vmask.any():
        return []

    hyps = generate_hypotheses(vmask, gray, cfg.geometry)
    scores = {}
    for hname, (hmask, reg, _) in hyps.items():
        if hmask is None:
            continue
        scores[hname] = hypothesis_score(gray, hmask, vmask, reg, cfg.geometry)

    if not scores:
        return []

    best_name = max(scores, key=scores.get)
    info[f"{name}_hypotheses"] = {k: round(v, 4) for k, v in scores.items()}
    info[f"{name}_chosen"] = best_name

    hmask, _, hquad = hyps[best_name]

    if best_name == "quad" and hquad is not None:
        q = hquad
        if cfg.geometry.snap_edges:
            q, notes = snap_quad_edges(q, gray,
                                        cfg.geometry.snap_band_frac,
                                        cfg.geometry.snap_min_grad)
            info[f"{name}_snap"] = notes
        aspect, how = pick_aspect(q, gray.shape[:2])
        info[f"{name}_aspect_method"] = how
        if how == "edge_lengths":
            flags.append(f"{name}:aspect_fallback_edge_lengths")
        res = warp_quad(img, vmask, q, aspect, cfg.geometry.warp_max_px)
        if res is not None:
            wi, wm, winfo = res
            info[f"{name}_warp"] = winfo
            alpha = alpha_from_mask(wm, cfg.geometry.edge_inset_px,
                                     cfg.geometry.feather_sigma)
            return save_variant(out_dir, name, wi, alpha, cfg.output.jpeg_quality)
        flags.append(f"{name}:warp_failed")

    alpha = alpha_from_mask(hmask, cfg.geometry.edge_inset_px,
                             cfg.geometry.feather_sigma)
    H, W = hmask.shape
    ys, xs = np.where(hmask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    p = int(cfg.output.pad_frac * max(x1 - x0, y1 - y0))
    x0, y0 = max(x0 - p, 0), max(y0 - p, 0)
    x1, y1 = min(x1 + p, W), min(y1 + p, H)
    return save_variant(out_dir, name,
                         img[y0:y1, x0:x1], alpha[y0:y1, x0:x1],
                         cfg.output.jpeg_quality)


def process_painting(img, gray, sam_predictor, art: dict, aux: dict,
                     out_dir: Path, pid: str, cfg: Config) -> dict:
    t0 = time.time()
    flags, info = [], {}
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_surface = art["mask"].copy()
    if not yolo_surface.any():
        return dict(painting=pid, status="failed",
                    flags=["empty_yolo_surface"],
                    seconds=round(time.time() - t0, 2))

    surface = refine_mask(img, yolo_surface, sam_predictor, art["box"], cfg)
    surface, _ = largest_component(surface)
    if not surface.any():
        surface = yolo_surface
        surface, _ = largest_component(surface)
        flags.append("refinement_empty_used_yolo")

    if cfg.content.enabled:
        cm = content_mask(img, surface, cfg.content)
        if cm is not None and cm.any():
            bs_yolo = boundary_support(gray, surface)
            bs_cm = boundary_support(gray, cm)
            ratio = np.count_nonzero(cm) / max(np.count_nonzero(surface), 1)
            info["content_mask_ratio"] = round(ratio, 3)
            info["content_support_yolo"] = round(bs_yolo, 3)
            info["content_support_content"] = round(bs_cm, 3)
            if (cfg.content.min_content_ratio <= ratio <= cfg.content.max_content_ratio
                    and bs_cm > bs_yolo + cfg.content.min_support_gain):
                surface = cm
                surface, _ = largest_component(surface)
                flags.append("used_content_mask_for_white_on_white")

    if "canvas_edge" in aux and aux["canvas_edge"].any():
        ce = aux["canvas_edge"]
        overlap = np.count_nonzero(surface & ce) / max(np.count_nonzero(surface), 1)
        if overlap > 0.7:
            flags.append(f"canvas_edge_contains_surface_{overlap:.2f}_skipped")
        else:
            cand = surface & ~ce
            cand, _ = largest_component(cand)
            if cand.any() and np.count_nonzero(cand) > 0.3 * np.count_nonzero(surface):
                surface = cand
            else:
                flags.append("canvas_edge_subtract_would_shrink_too_much")

    info["surface_area_px"] = int(np.count_nonzero(surface))

    variants = [("artwork_surface", surface)]
    has_mat = "mat" in aux and aux["mat"].any()
    has_frame = "frame" in aux and aux["frame"].any()
    if has_mat:
        v = surface | aux["mat"]
        v, _ = largest_component(v)
        if v.any():
            variants.append(("artwork_surface_mat", v))
        if has_frame:
            v2 = surface | aux["mat"] | aux["frame"]
            v2, _ = largest_component(v2)
            if v2.any():
                variants.append(("artwork_surface_mat_frame", v2))
    elif has_frame:
        v = surface | aux["frame"]
        v, _ = largest_component(v)
        if v.any():
            variants.append(("artwork_surface_frame", v))
    info["variants_emitted"] = [v[0] for v in variants]

    save_mask(out_dir, "artwork_surface", surface, cfg.output.jpeg_quality)
    save_mask(out_dir, "canvas_edge",
              aux.get("canvas_edge", np.zeros_like(surface)), cfg.output.jpeg_quality)
    save_mask(out_dir, "mat",
              aux.get("mat", np.zeros_like(surface)), cfg.output.jpeg_quality)
    save_mask(out_dir, "frame",
              aux.get("frame", np.zeros_like(surface)), cfg.output.jpeg_quality)

    all_files = []
    for vname, vmask in variants:
        try:
            files = render_variant(img, gray, vmask, vname, out_dir,
                                    cfg, flags, info)
            all_files.extend(files)
        except Exception as e:
            log.exception("%s: variant %s failed", pid, vname)
            flags.append(f"{vname}:render_failed:{e}")

    status = "needs_review" if flags else "success"
    return dict(painting=pid, yolo_confidence=art["conf"], status=status,
                flags=flags, info=info, files=all_files,
                seconds=round(time.time() - t0, 2))


def assign_aux(arts, aux_dets, shape):
    H, W = shape
    res = [{} for _ in arts]
    if not arts:
        return res
    zones = []
    for a in arts:
        x, y, w, h = cv2.boundingRect(a["mask"])
        k = int(0.25 * max(w, h)) | 1
        zones.append((cv2.dilate(a["mask"], np.ones((k, k), np.uint8)),
                      (x + w / 2, y + h / 2)))
    for cname, dets in aux_dets.items():
        for d in dets:
            best, bs = None, 0.0
            for i, (zone, (cx, cy)) in enumerate(zones):
                f = np.count_nonzero((d["mask"] > 0) & (zone > 0)) / \
                    max(np.count_nonzero(d["mask"]), 1)
                if f > bs:
                    best, bs = i, f
            if best is not None and bs >= 0.15:
                res[best][cname] = res[best].get(
                    cname, np.zeros(shape, np.uint8)) | d["mask"]
    return res


def process_image(model, sam_predictor, path: Path,
                  out_root: Path, cfg: Config) -> dict:
    img = read_image(path)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets = yolo_predict(model, img, cfg)
    names_rev = {v: k for k, v in model.names.items()}

    def _cid(name):
        if name in names_rev:
            return names_rev[name]
        for n, i in names_rev.items():
            if name.lower() in n.lower():
                return i
        return None

    surf_id = _cid(cfg.classes.surface)
    edge_id = _cid(cfg.classes.canvas_edge)
    frame_id = _cid(cfg.classes.frame)
    mat_id = _cid(cfg.classes.mat)

    surfaces = dedupe([d for d in dets if d["cls"] == surf_id]) if surf_id is not None else []
    surfaces.sort(key=lambda d: (round(d["box"][1] / (0.1 * H)), d["box"][0]))

    aux_dets = {
        "canvas_edge": [d for d in dets if d["cls"] == edge_id] if edge_id is not None else [],
        "frame": [d for d in dets if d["cls"] == frame_id] if frame_id is not None else [],
        "mat": [d for d in dets if d["cls"] == mat_id] if mat_id is not None else [],
    }

    img_dir = unique_dir(out_root / path.stem)
    shutil.copy2(path, img_dir / f"00_original{path.suffix.lower()}")

    manifest = dict(input=path.name, size=[W, H],
                    n_paintings=len(surfaces), paintings=[])
    if not surfaces:
        manifest["status"] = "needs_review: no artwork_surface detected"
        (img_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=jdefault), encoding="utf-8")
        return manifest

    aux_assign = assign_aux(surfaces, aux_dets, (H, W))
    for i, (a, aux) in enumerate(zip(surfaces, aux_assign), 1):
        pid = f"painting_{i:02d}"
        sub = img_dir / pid
        try:
            manifest["paintings"].append(
                process_painting(img, gray, sam_predictor, a, aux, sub, pid, cfg))
        except Exception as e:
            log.exception("%s %s failed", path.name, pid)
            manifest["paintings"].append(
                dict(painting=pid, status="failed", flags=[str(e)]))

    (img_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=jdefault), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sam2-checkpoint", default=None)
    ap.add_argument("--sam2-config", default=None)
    ap.add_argument("--no-sam2", action="store_true")
    ap.add_argument("--no-grabcut", action="store_true")
    ap.add_argument("--no-content", action="store_true",
                    help="Disable white-on-white content detection")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model: cfg.paths.model_path = args.model
    if args.input: cfg.paths.input_dir = args.input
    if args.output: cfg.paths.output_dir = args.output
    if args.device is not None: cfg.yolo.device = args.device
    if args.sam2_checkpoint: cfg.paths.sam2_checkpoint = args.sam2_checkpoint
    if args.sam2_config: cfg.paths.sam2_config = args.sam2_config
    if args.no_sam2: cfg.sam2.enabled = False
    if args.no_grabcut: cfg.refine.grabcut_enabled = False
    if args.no_content: cfg.content.enabled = False

    out_root = Path(cfg.paths.output_dir)
    setup_logging(out_root)
    log.info("Config: %s", json.dumps(asdict(cfg), indent=2, default=jdefault))
    log.info("Codecs: HEIC=%s AVIF=%s", _HEIF_OK, _AVIF_OK)

    model = load_yolo(cfg.paths.model_path)
    sam_predictor = None
    if cfg.sam2.enabled:
        sam_predictor = load_sam2(cfg)
    if sam_predictor is None and cfg.refine.grabcut_enabled:
        log.info("Using grabCut refinement (OpenCV-only)")
    elif sam_predictor is None:
        log.warning("No refinement — using raw YOLO masks")

    in_dir = Path(cfg.paths.input_dir)
    if not in_dir.exists():
        sys.exit(f"Input dir missing: {in_dir}")
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in cfg.output.exts)
    if not files:
        sys.exit(f"No images in {in_dir}")

    log.info("Processing %d images", len(files))
    rows, times = [], []
    for p in tqdm(files, desc="Processing"):
        t = time.time()
        try:
            manifest = process_image(model, sam_predictor, p, out_root, cfg)
            paints = manifest.get("paintings") or [manifest]
            for paint in paints:
                rows.append(dict(
                    image=p.name,
                    painting=paint.get("painting", ""),
                    status=paint.get("status", manifest.get("status", "")),
                    variants=";".join(paint.get("info", {}).get("variants_emitted", [])),
                    chosen_shape=paint.get("info", {}).get("artwork_surface_chosen", ""),
                    flags=";".join(paint.get("flags", [])),
                    seconds=paint.get("seconds", "")))
        except Exception as e:
            log.exception("Failed: %s", p.name)
            rows.append(dict(image=p.name, painting="", status="failed",
                             variants="", chosen_shape="",
                             flags=str(e), seconds=""))
        times.append(time.time() - t)

    if rows:
        with open(out_root / "batch_summary.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    (out_root / "batch_summary.json").write_text(
        json.dumps(dict(
            config=asdict(cfg), n_images=len(files), n_paintings=len(rows),
            avg_time_per_image_s=round(float(np.mean(times)), 2) if times else None,
            status_counts={s: sum(r["status"] == s for r in rows)
                           for s in ("success", "needs_review", "failed")},
            rows=rows), indent=2, default=jdefault),
        encoding="utf-8")

    log.info("Done. Outputs in %s", out_root)


if __name__ == "__main__":
    main()