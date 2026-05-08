"""
Pot 检测结果与分割结果融合（Task 逻辑）。

约束：
- 画面中至多一个 pot。
- 若检测框（bbox）相对分割区域更靠下，则以分割区域外接框（可带边距）作为最终 bbox。
- pot 不应出现在画面下方三分之一；若 pot 的 bbox 与下方三分之一带的 IoU > 阈值，则忽略该 pot。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 与 train_yolo12_seg / labelme 中 CLASS_NAMES 一致：0 container, 1 food, 2 pot, 3 slices
POT_CLASS_ID = 2

# 与下方三分之一带的 IoU 超过该值则丢弃（用户要求 20%）
BOTTOM_THIRD_IOU_THRESHOLD = 0.20

# 用分割区域替代 bbox 时，对外接框的各边扩展比例（相对 max(宽,高)）
SEG_SURROUND_PAD_RATIO = 0.02


@dataclass
class PotFusedResult:
    """融合后的单个 pot 结果；ignored=True 表示应丢弃。"""

    xyxy: Optional[Tuple[float, float, float, float]]
    ignored: bool
    reason: str = ""


def iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """轴对齐框 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def bottom_third_xyxy(img_h: int, img_w: int) -> Tuple[float, float, float, float]:
    """画面下方三分之一矩形 [y 从 2/3 * H 到 H]。"""
    y0 = (2.0 / 3.0) * float(img_h)
    return (0.0, y0, float(img_w), float(img_h))


def center_y(xyxy: Tuple[float, float, float, float]) -> float:
    return (xyxy[1] + xyxy[3]) / 2.0


def clamp_xyxy(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = float(np.clip(x1, 0, img_w))
    x2 = float(np.clip(x2, 0, img_w))
    y1 = float(np.clip(y1, 0, img_h))
    y2 = float(np.clip(y2, 0, img_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def bbox_from_binary_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    mask: HxW，非零为前景。返回 xyxy；若无前景则 None。
    """
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask != 0)
    if ys.size == 0:
        return None
    y1, y2 = float(ys.min()), float(ys.max())
    x1, x2 = float(xs.min()), float(xs.max())
    return (x1, y1, x2, y2)


def surround_bbox(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    pad_ratio: float = SEG_SURROUND_PAD_RATIO,
) -> Tuple[float, float, float, float]:
    """在分割外接框基础上按「周边」扩展。"""
    x1, y1, x2, y2 = xyxy
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    pad = pad_ratio * max(w, h)
    return clamp_xyxy((x1 - pad, y1 - pad, x2 + pad, y2 + pad), img_w, img_h)


def should_ignore_bottom_third(
    pot_xyxy: Tuple[float, float, float, float],
    img_h: int,
    img_w: int,
    threshold: float = BOTTOM_THIRD_IOU_THRESHOLD,
) -> bool:
    """pot bbox 与下方三分之一带的 IoU 超过 threshold 则 True（应忽略）。"""
    bt = bottom_third_xyxy(img_h, img_w)
    return iou_xyxy(pot_xyxy, bt) > threshold


def fuse_pot_bbox_and_seg(
    det_xyxy: Optional[Tuple[float, float, float, float]],
    seg_xyxy: Optional[Tuple[float, float, float, float]],
    img_h: int,
    img_w: int,
    pad_ratio: float = SEG_SURROUND_PAD_RATIO,
) -> Optional[Tuple[float, float, float, float]]:
    """
    至多一个 pot。

    - 仅有 det：用 det。
    - 仅有 seg：用 seg 的周边外接框。
    - 二者都有：若 det 的中心 y 大于 seg 外接框中心 y（检测框在分割区域「下面」），
      则以分割外接框 + 周边作为最终 bbox；否则使用 det。
    """
    if det_xyxy is None and seg_xyxy is None:
        return None
    if seg_xyxy is None:
        return clamp_xyxy(det_xyxy, img_w, img_h)
    if det_xyxy is None:
        return surround_bbox(seg_xyxy, img_w, img_h, pad_ratio)

    det_cy = center_y(det_xyxy)
    seg_cy = center_y(seg_xyxy)
    if det_cy > seg_cy:
        return surround_bbox(seg_xyxy, img_w, img_h, pad_ratio)
    return clamp_xyxy(det_xyxy, img_w, img_h)


def resolve_single_pot(
    det_xyxy: Optional[Tuple[float, float, float, float]],
    seg_xyxy: Optional[Tuple[float, float, float, float]],
    img_h: int,
    img_w: int,
) -> PotFusedResult:
    """
    完整流程：融合 bbox/seg，再应用下方三分之一规则。
    """
    fused = fuse_pot_bbox_and_seg(det_xyxy, seg_xyxy, img_h, img_w)
    if fused is None:
        return PotFusedResult(xyxy=None, ignored=False, reason="no_pot")

    if should_ignore_bottom_third(fused, img_h, img_w):
        return PotFusedResult(xyxy=None, ignored=True, reason="bottom_third_iou")

    return PotFusedResult(xyxy=fused, ignored=False, reason="")


# --- 与 Ultralytics YOLO 结果对接的辅助函数 ---------------------------------

def _xyxy_from_ultralytics_box(box) -> Tuple[float, float, float, float]:
    t = box.xyxy[0].cpu().numpy()
    return (float(t[0]), float(t[1]), float(t[2]), float(t[3]))


def pick_single_pot_from_result(result, pot_class_id: int = POT_CLASS_ID):
    """
    从 ultralytics.engine.results.Results 中取置信度最高的一个 pot 检测框与对应 mask。

    返回:
        det_xyxy, seg_mask (HxW numpy 或 None), conf
        若无 pot 类别检测则 (None, None, None)
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None

    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    best_i: Optional[int] = None
    best_c = -1.0
    for i in range(len(cls)):
        if cls[i] != pot_class_id:
            continue
        if conf[i] > best_c:
            best_c = float(conf[i])
            best_i = i

    if best_i is None:
        return None, None, None

    det_xyxy = _xyxy_from_ultralytics_box(boxes[best_i])

    seg_mask = None
    if result.masks is not None and len(result.masks) > best_i:
        m = result.masks.data[best_i].cpu().numpy()
        # 若尺寸与原图不一致，此处假设为模型输入尺寸上的 mask；调用方需先 result.masks.xy 或按原图 resize
        seg_mask = m

    return det_xyxy, seg_mask, best_c


def pot_seg_xyxy_from_mask(
    seg_mask: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    将模型输出的 mask（可能与原图不同尺寸）resize 到原图再取外接框。
    """
    try:
        import cv2
    except ImportError:
        from PIL import Image

        if seg_mask.shape[:2] != (orig_h, orig_w):
            im = Image.fromarray((seg_mask > 0.5).astype(np.uint8) * 255)
            im = im.resize((orig_w, orig_h), resample=Image.NEAREST)
            seg_mask = np.array(im) > 127
        else:
            seg_mask = seg_mask > 0.5
    else:
        if seg_mask.shape[:2] != (orig_h, orig_w):
            seg_mask = cv2.resize(
                seg_mask.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        seg_mask = seg_mask > 0.5

    return bbox_from_binary_mask(seg_mask.astype(np.uint8))


def process_frame_pot(
    result,
    orig_h: Optional[int] = None,
    orig_w: Optional[int] = None,
    pot_class_id: int = POT_CLASS_ID,
) -> PotFusedResult:
    """
    单帧：从 YOLO segment 的 result 得到融合后的 pot（至多一个）。

    orig_h, orig_w: 原始图像高宽；省略时使用 result.orig_shape（用于底部三分之一与 mask 对齐）。
    """
    if orig_h is None or orig_w is None:
        h, w = result.orig_shape[:2]
        orig_h, orig_w = int(h), int(w)

    det_xyxy, seg_mask, _conf = pick_single_pot_from_result(result, pot_class_id)

    seg_xyxy = None
    if seg_mask is not None:
        seg_xyxy = pot_seg_xyxy_from_mask(seg_mask, orig_h, orig_w)

    return resolve_single_pot(det_xyxy, seg_xyxy, orig_h, orig_w)
