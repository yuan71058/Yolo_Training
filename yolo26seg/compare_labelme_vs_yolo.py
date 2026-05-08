# -*- coding: utf-8 -*-
"""
从 seg_dataset/train 抽一张图+json，从 yolo12_runs 找同图+转换后的 txt，
生成对比预览：左侧为 labelme 原标注，右侧为 YOLO 转换结果；并可用 labelme 打开原图、用系统查看器打开预览图。
"""
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# 配置
SEG_TRAIN = Path(r"E:\train_pot_1105_2\seg_dataset\train")
YOLO_ROOT = Path(r"E:\yolo12_runs")
COMPARE_OUT = Path(r"E:\train_pot_1105_2\compare_labelme_vs_yolo")
CLASS_NAMES = ["container", "food", "pot", "slices"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]  # BGR


def _pick_one_pair(seg_train: Path):
    """从 seg_dataset/train 取第一个「有 json」的 (img_path, json_path)。"""
    for fp in sorted(seg_train.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
            continue
        j = seg_train / f"{fp.stem}.json"
        if j.exists():
            return fp, j
    return None, None


def _find_yolo_pair(yolo_root: Path, stem: str, ext: str):
    """在 yolo12_runs 的 images/train、images/val 与 labels 下找同名图片和 txt。"""
    name = stem + ext
    for split in ("train", "val"):
        img_path = yolo_root / "images" / split / name
        if img_path.exists():
            txt_path = yolo_root / "labels" / split / f"{stem}.txt"
            if txt_path.exists():
                return img_path, txt_path
    return None, None


def _draw_labelme(img_bgr, json_path: Path, label_colors=None):
    """在图上绘制 labelme JSON 的 polygon/mask 轮廓（points 为像素坐标）。"""
    if label_colors is None:
        label_colors = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h, w = img_bgr.shape[:2]
    for sh in data.get("shapes", []):
        label = sh.get("label", "")
        if label not in label_colors:
            idx = len(label_colors) % len(COLORS)
            label_colors[label] = COLORS[idx]
        pts = sh.get("points", [])
        if not pts or len(pts) < 2:
            continue
        # 确保为整数且不越界，避免绘制错位
        pts_np = np.array([[float(p[0]), float(p[1])] for p in pts], dtype=np.float32)
        pts_np = np.round(pts_np).astype(np.int32)
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, w - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, h - 1)
        cv2.polylines(img_bgr, [pts_np], True, label_colors[label], 2)
        x0, y0 = int(pts_np[0][0]), int(pts_np[0][1])
        cv2.putText(img_bgr, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_colors[label], 1)
    return img_bgr


def _draw_yolo(img_bgr, txt_path: Path, label_colors=None):
    """在图上绘制 YOLO segment txt。正常为归一化 0~1；若数值均>1 则按像素坐标解析（旧 bug 生成的 txt）。"""
    if label_colors is None:
        label_colors = {}
    h, w = img_bgr.shape[:2]
    if not txt_path.exists():
        return img_bgr
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            cid = int(parts[0])
            raw_x = [float(parts[i]) for i in range(1, len(parts), 2)]
            raw_y = [float(parts[i]) for i in range(2, len(parts), 2)]
        except (ValueError, IndexError):
            continue
        if len(raw_x) < 3 or len(raw_y) < 3:
            continue
        # 若最大值 > 1.5 则视为旧版误写的像素坐标，不再乘 w/h
        all_vals = raw_x + raw_y
        if max(all_vals) > 1.5:
            xs, ys = raw_x, raw_y
        else:
            xs = [x * w for x in raw_x]
            ys = [y * h for y in raw_y]
        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        if label not in label_colors:
            label_colors[label] = COLORS[cid % len(COLORS)]
        pts = np.array(list(zip(xs, ys)), dtype=np.int32)
        cv2.polylines(img_bgr, [pts], True, label_colors[label], 2)
        x0, y0 = int(xs[0]), int(ys[0])
        cv2.putText(img_bgr, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_colors[label], 1)
    return img_bgr


def main():
    if cv2 is None:
        print("需要 opencv-python: pip install opencv-python")
        sys.exit(1)
    seg_train = Path(SEG_TRAIN)
    yolo_root = Path(YOLO_ROOT)
    out_dir = Path(COMPARE_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path, json_path = _pick_one_pair(seg_train)
    if img_path is None:
        print(f"在 {seg_train} 下未找到「图片+同名 json」")
        sys.exit(1)
    stem, ext = img_path.stem, img_path.suffix
    yolo_img, yolo_txt = _find_yolo_pair(yolo_root, stem, ext)
    if yolo_img is None or yolo_txt is None:
        print(f"在 {yolo_root} 下未找到同名图片或 txt: {stem}{ext}")
        sys.exit(1)

    # 用同一张原图（从 seg 或 yolo 读均可，保持一致）
    img = cv2.imread(str(img_path))
    if img is None:
        img = cv2.imread(str(yolo_img))
    if img is None:
        print("无法读取图片")
        sys.exit(1)
    h, w = img.shape[:2]

    # 左侧：labelme 原标注
    left = img.copy()
    _draw_labelme(left, json_path)
    # 右侧：YOLO 转换结果（若 txt 里是像素坐标会按像素绘制）
    right = img.copy()
    _draw_yolo(right, yolo_txt)
    # 若检测到 txt 为像素坐标（旧转换 bug），提示重跑转换
    try:
        first = next(iter(yolo_txt.read_text(encoding="utf-8").strip().splitlines()), "")
        vals = [float(x) for x in first.split()[1:] if first.split()]
        if vals and max(vals) > 1.5:
            print("[WARN] 当前 txt 似为像素坐标（非 0~1），请重新运行 labelme_to_yolo_segments.py 生成正确归一化 txt 后再对比。")
    except Exception:
        pass

    # 拼成左右对比图（若图太宽可缩小）
    max_h = 720
    if h > max_h:
        scale = max_h / h
        left = cv2.resize(left, None, fx=scale, fy=scale)
        right = cv2.resize(right, None, fx=scale, fy=scale)
    pad = 20
    canvas = np.hstack([left, np.ones((left.shape[0], pad, 3), dtype=np.uint8) * 200, right])
    # 标题
    cv2.putText(canvas, "Labelme (original)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "YOLO (converted)", (left.shape[1] + pad + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    compare_path = out_dir / f"{stem}_compare.png"
    cv2.imwrite(str(compare_path), canvas)
    print(f"[OK] 对比图已保存: {compare_path}")

    # 复制原图+json 到对比目录，并写入 imagePath + imageData，避免 labelme 报错 'imageData'
    import shutil
    orig_img_out = out_dir / f"{stem}_original{ext}"
    orig_json_out = out_dir / f"{stem}_original.json"
    shutil.copy2(img_path, orig_img_out)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["imagePath"] = orig_img_out.name
    # labelme 打开时依赖 imageData，必须写入有效 base64，否则会报错 'imageData'
    with open(orig_img_out, "rb") as f:
        data["imageData"] = base64.b64encode(f.read()).decode("ascii")
    with open(orig_json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 原图与 JSON 已复制到: {orig_img_out}, {orig_json_out}（已写入 imageData 供 labelme 使用）")

    # 用系统默认程序打开对比图
    if sys.platform == "win32":
        os.startfile(compare_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(compare_path)], check=False)
    else:
        subprocess.run(["xdg-open", str(compare_path)], check=False)

    # 尝试用 labelme 打开原图（可选）
    labelme_cmd = None
    for cmd in ("labelme", "labelme.exe"):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
            if r.returncode == 0 or b"labelme" in (r.stdout or b"") or b"labelme" in (r.stderr or b""):
                labelme_cmd = cmd
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    if labelme_cmd:
        labels_file = out_dir / "labels.txt"
        labels_file.write_text("\n".join(CLASS_NAMES), encoding="utf-8")
        subprocess.Popen([labelme_cmd, str(orig_img_out), "--labels", str(labels_file)], cwd=str(out_dir))
        print(f"[INFO] 已用 labelme 打开原图: {orig_img_out}")
    else:
        print("[INFO] 未检测到 labelme，请手动用 labelme 打开:", orig_img_out)

    print("\n对比说明: 左侧=Labelme 原标注（仅含 points 的形状），右侧=转换后的 YOLO segment。")
    print("若原标注含 brush/mask 无 points，请用已打开的 labelme 窗口看完整原图。")


if __name__ == "__main__":
    main()
