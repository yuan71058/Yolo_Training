# -*- coding: utf-8 -*-
"""
将 labelme 标注的 JSON 转为 YOLOv12（及 Ultralytics）segments 训练格式。
- 输入：E:\\train_pot_1105_2\\seg_dataset\\train 下已标注的图片及其同名 .json（原 JSON 保留）
- 输出：按 8:2 划分到 E:/yolo12_runs 的 images/train、images/val 与 labels/train、labels/val
- 支持 labelme 的 polygon、rectangle；若为 mask 形态则尝试从 points 或解码 mask 得到轮廓后转换
"""
import json
import random
import shutil
from pathlib import Path

import numpy as np

# 可选：mask 解码为轮廓时用到
try:
    import cv2
except ImportError:
    cv2 = None

# ------------------------- 配置 -------------------------
SOURCE_DIR = Path(r"E:\train_pot_1105_2\seg_dataset\train")
OUT_ROOT = Path(r"E:/yolo12_runs")
TRAIN_RATIO = 0.8
SEED = 42

# 类别顺序（与 labelme 中 label 一致，索引即 class_id）
CLASS_NAMES = ["container", "food", "pot", "slices"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# -------------------------


def _get_image_size(img_path: Path) -> tuple:
    """从图片文件读取 (width, height)。"""
    if cv2 is not None:
        im = cv2.imread(str(img_path))
        if im is not None:
            return im.shape[1], im.shape[0]
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]
    except Exception:
        pass
    return 1, 1


def _shape_to_normalized_points(shape: dict, img_w: int, img_h: int) -> list:
    """
    将 labelme 的一个 shape 转为 YOLO 所需的归一化多边形列表，每个多边形为 [x1,y1,x2,y2,...]（0~1）。
    返回 [[poly1], [poly2], ...]，一个 shape 可对应多块区域（如 mask 内多块不连通区域）；空列表表示无法转换。
    """
    if img_w <= 0 or img_h <= 0:
        return []
    pts = shape.get("points")
    shape_type = (shape.get("shape_type") or "polygon").lower()

    def _norm_one(points_xy):
        out = []
        for x, y in points_xy:
            out.append(max(0.0, min(1.0, x / img_w)))
            out.append(max(0.0, min(1.0, y / img_h)))
        return out

    # polygon：直接使用 points
    if shape_type == "polygon" and pts and len(pts) >= 3:
        return [_norm_one([(float(p[0]), float(p[1])) for p in pts])]

    # rectangle：两点转四个角点（左上、右上、右下、左下）
    if shape_type == "rectangle" and pts and len(pts) >= 2:
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return [_norm_one(corners)]

    # circle：圆心+圆上一点，近似为多边形（约 32 点）
    if shape_type == "circle" and pts and len(pts) >= 2:
        cx, cy = float(pts[0][0]), float(pts[0][1])
        rx = ((pts[1][0] - cx) ** 2 + (pts[1][1] - cy) ** 2) ** 0.5
        ry = rx
        n_pts = 32
        out = []
        for i in range(n_pts):
            t = 2 * np.pi * i / n_pts
            out.append(max(0.0, min(1.0, (cx + rx * np.cos(t)) / img_w)))
            out.append(max(0.0, min(1.0, (cy + ry * np.sin(t)) / img_h)))
        return [out]

    # 有 points 的其他类型（如 point、line 等）尽量当折线用（至少 3 点才成面）
    if pts and len(pts) >= 3:
        return [_norm_one([(float(p[0]), float(p[1])) for p in pts])]

    # mask：labelme 的 brush/mask 可能存为 (1) points 轮廓 (2) base64 PNG 图；PNG 内可能有多块不连通区域，每块输出一条
    if shape_type == "mask":
        if pts and len(pts) >= 3:
            return [_norm_one([(float(p[0]), float(p[1])) for p in pts])]
        # 编码的 mask：解码出所有轮廓，每块区域一条多边形
        mask_data = shape.get("mask") or shape.get("mask_encoding")
        if cv2 is not None and mask_data is not None and pts and len(pts) >= 2:
            xs_pt = [float(p[0]) for p in pts]
            ys_pt = [float(p[1]) for p in pts]
            x1, x2 = min(xs_pt), max(xs_pt)
            y1, y2 = min(ys_pt), max(ys_pt)
            bbox_xyxy = (x1, y1, x2, y2)
            contours_list = _decode_mask_to_contours(mask_data, img_w, img_h, bbox_xyxy=bbox_xyxy)
            if contours_list:
                result = []
                for contour_xy in contours_list:
                    if len(contour_xy) < 3:
                        continue
                    result.append(_norm_one(contour_xy))
                return result
    return []


def _decode_mask_to_contours(mask_data, img_w: int, img_h: int, bbox_xyxy=None):
    """
    将 labelme 的 mask 转为多个轮廓的列表，每个轮廓为 [(x,y),...]（图像坐标系）。
    只保留面积最大的至多 3 个区域，且第 2、3 个面积不小于最大区域的 10%，避免点缀小碎片被误保留。
    bbox_xyxy: (x1,y1,x2,y2) 表示 mask 小图在原图中的裁剪框；若提供则把轮廓从 mask 像素坐标映射到图像坐标。
    """
    if cv2 is None:
        return []
    import base64
    import io
    raw_b64 = None
    if isinstance(mask_data, str):
        raw_b64 = mask_data.strip()
    elif isinstance(mask_data, dict):
        raw_b64 = mask_data.get("data") or mask_data.get("data_encoded")
        if isinstance(raw_b64, bytes):
            raw_b64 = None
        elif raw_b64 is not None and not isinstance(raw_b64, str):
            raw_b64 = None
    if not raw_b64:
        return []
    try:
        raw_b64 = raw_b64.replace("\n", "").replace("\r", "")
        buf = base64.b64decode(raw_b64)
    except Exception:
        return []
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(buf))
        mask = np.array(img)
    except Exception:
        try:
            mask = np.frombuffer(buf, dtype=np.uint8)
            if mask.size == img_h * img_w:
                mask = mask.reshape((img_h, img_w))
            else:
                return []
        except Exception:
            return []
    if mask is None or getattr(mask, "size", 0) == 0:
        return []
    if len(mask.shape) == 3:
        mask = mask.max(axis=2) if mask.shape[2] > 1 else mask[:, :, 0]
    h_mask, w_mask = mask.shape[:2]
    img_area = h_mask * w_mask
    if img_area == 0:
        return []

    # 二值化，并收集「物体」轮廓（可能多块不连通）
    mask_pos = np.uint8(mask > 0)
    contours_pos, _ = cv2.findContours(mask_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 若正相最大轮廓接近整图则视为背景，改用反相
    use_neg = False
    if contours_pos:
        c_largest = max(contours_pos, key=cv2.contourArea)
        if cv2.contourArea(c_largest) >= 0.9 * img_area:
            use_neg = True
    if use_neg:
        if mask.max() <= 1:
            mask_neg = np.uint8(1 - mask_pos)
        else:
            mask_neg = np.uint8(255 - mask)
        contours_pos, _ = cv2.findContours(mask_neg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 只保留最大的至多 3 个区域，且第 2、3 个面积不小于最大区域的 10%，避免点缀小碎片被误保留
    min_area = 0.01 * img_area
    candidates = []
    for c in contours_pos:
        if c is None:
            continue
        area = cv2.contourArea(c)
        if area < min_area or area >= 0.9 * img_area:
            continue
        if len(c) < 3:
            continue
        candidates.append((area, c))
    candidates.sort(key=lambda x: x[0], reverse=True)
    # 最多取 3 个，第 2、3 个面积须 >= 最大面积的 10%
    max_keep = 3
    min_ratio = 0.10  # 第二、三个区域不小于第一个的 10%
    kept = []
    for i, (area, c) in enumerate(candidates):
        if i >= max_keep:
            break
        if i > 0 and candidates[0][0] > 0 and area < candidates[0][0] * min_ratio:
            break
        pts = c.reshape(-1, 2).tolist()
        # 从 mask 像素坐标映射到图像坐标
        if bbox_xyxy is not None and len(bbox_xyxy) == 4 and w_mask > 0 and h_mask > 0:
            x1, y1, x2, y2 = bbox_xyxy
            scale_x = (x2 - x1) / w_mask
            scale_y = (y2 - y1) / h_mask
            pts = [(x1 + x * scale_x, y1 + y * scale_y) for x, y in pts]
        elif (w_mask, h_mask) != (img_w, img_h) and w_mask > 0 and h_mask > 0:
            scale_x = img_w / w_mask
            scale_y = img_h / h_mask
            pts = [(x * scale_x, y * scale_y) for x, y in pts]
        kept.append(pts)
    return kept


def labelme_json_to_yolo_lines(json_path: Path, img_w: int, img_h: int) -> list:
    """
    读入单个 labelme JSON，返回 YOLO segment 行列表，每行为 "class_id x1 y1 x2 y2 ..."（归一化 0~1）。
    始终使用传入的 img_w/img_h（必须来自实际读图）做归一化，不采用 JSON 内宽高，避免与真实尺寸不一致导致错位。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if img_w <= 0 or img_h <= 0:
        return []
    lines = []
    for sh in data.get("shapes", []):
        label = (sh.get("label") or "").strip()
        if label not in CLASS_TO_ID:
            continue
        class_id = CLASS_TO_ID[label]
        polygons = _shape_to_normalized_points(sh, img_w, img_h)
        for norm in polygons:
            if not norm or len(norm) < 6:
                continue
            line = f"{class_id} " + " ".join(f"{x:.6f}" for x in norm)
            lines.append(line)
    return lines


def main():
    source_dir = Path(SOURCE_DIR)
    out_root = Path(OUT_ROOT)
    if not source_dir.is_dir():
        print(f"[ERROR] 源目录不存在: {source_dir}")
        return

    # 收集所有「有 json」的图片
    pairs = []
    for fp in sorted(source_dir.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
            continue
        json_path = source_dir / f"{fp.stem}.json"
        if not json_path.exists():
            continue
        pairs.append((fp, json_path))

    if not pairs:
        print(f"[WARN] 在 {source_dir} 下未找到任何「图片+同名 json」对")
        return

    # 8:2 划分
    random.seed(SEED)
    random.shuffle(pairs)
    n_train = max(1, int(len(pairs) * TRAIN_RATIO))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    dirs = {
        "train": (out_root / "images" / "train", out_root / "labels" / "train"),
        "val": (out_root / "images" / "val", out_root / "labels" / "val"),
    }
    for split_name, (img_dir, lab_dir) in dirs.items():
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

    def process_split(split_pairs, split_name):
        img_dir, lab_dir = dirs[split_name]
        for img_path, json_path in split_pairs:
            # 归一化必须用真实图片尺寸，避免 JSON 内 imageWidth/Height 与文件不一致导致右侧预览错位/乱线
            img_w, img_h = _get_image_size(img_path)
            if img_w <= 0 or img_h <= 0:
                continue

            yolo_lines = labelme_json_to_yolo_lines(json_path, img_w, img_h)
            if not yolo_lines:
                continue
            # 复制图片（原路径不动，仅复制到 yolo 目录）
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            txt_path = lab_dir / f"{img_path.stem}.txt"
            txt_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")

    # 生成 data.yaml 便于 YOLO 训练
    nc = len(CLASS_NAMES)
    yaml_path = out_root / "data.yaml"
    yaml_content = f"""# YOLO segment dataset (from labelme)
path: {out_root.resolve()}
train: images/train
val: images/val

nc: {nc}
names: {CLASS_NAMES}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"[OK] 已转换 {len(pairs)} 条（train {len(train_pairs)} / val {len(val_pairs)}）")
    print(f"     输出: {out_root}")
    print(f"     图片与标签: images/{{train,val}}, labels/{{train,val}}")
    print(f"     原 JSON 未改动，仍位于: {source_dir}")
    print(f"     data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
