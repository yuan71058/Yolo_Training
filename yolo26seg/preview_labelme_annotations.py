# -*- coding: utf-8 -*-
"""
预览外包标注：读取图片与同名 labelme JSON，叠加显示多边形标注。
支持两种结构（只要指定 dataset 根目录即可）：
  1) 同目录：图片与 JSON 在同一目录，除后缀外文件名相同。
  2) 分目录：dataset/images（或 image）+ dataset/labels（或 label），文件名一一对应。
用法:
  python preview_labelme_annotations.py E:\\dataset
  python preview_labelme_annotations.py --dir E:\\path --save E:\\previews
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 可选依赖：用于显示与保存
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# 可能的「图片目录名」与「标签目录名」组合（小写），优先 images/labels
IMAGE_DIR_NAMES = ("images", "image")
LABEL_DIR_NAMES = ("labels", "label")


def _find_image_label_dirs(root: Path):
    """
    若 dataset 为「image + label 分目录」结构，返回 (images_dir, labels_dir)；
    否则返回 (None, None)。
    """
    root = Path(root)
    if not root.is_dir():
        return None, None
    for img_name in IMAGE_DIR_NAMES:
        img_dir = root / img_name
        if not img_dir.is_dir():
            continue
        for lab_name in LABEL_DIR_NAMES:
            lab_dir = root / lab_name
            if lab_dir.is_dir():
                return img_dir, lab_dir
    return None, None


def _pairs_from_image_label_dirs(img_dir: Path, lab_dir: Path):
    """从 images 目录与 labels 目录按文件名匹配，返回 [(img_path, json_path), ...]。"""
    pairs = []
    for fp in sorted(img_dir.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
            continue
        json_path = lab_dir / f"{fp.stem}.json"
        if json_path.exists():
            pairs.append((fp, json_path))
    return pairs


def find_image_json_pairs(root: Path):
    """
    扫描目录，返回 [(image_path, json_path), ...]。
    - 若存在 dataset/images + dataset/labels（或 image/label），则按文件名在两边匹配。
    - 若存在 dataset/train、dataset/val 等子目录且子目录内各有 images+labels，则汇总所有子目录的配对。
    - 否则在同一目录下找「图片 + 同名 .json」。
    """
    root = Path(root)
    if not root.is_dir():
        return []

    img_dir, lab_dir = _find_image_label_dirs(root)
    if img_dir is not None and lab_dir is not None:
        return _pairs_from_image_label_dirs(img_dir, lab_dir)

    # 尝试子目录（如 train、val）内部分别为 images+labels
    all_pairs = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        si, sl = _find_image_label_dirs(sub)
        if si is not None and sl is not None:
            all_pairs.extend(_pairs_from_image_label_dirs(si, sl))
    if all_pairs:
        return all_pairs

    # 同目录结构
    pairs = []
    for fp in sorted(root.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
            continue
        json_path = root / f"{fp.stem}.json"
        if json_path.exists():
            pairs.append((fp, json_path))
    return pairs


def load_labelme_shapes(json_path: Path):
    """解析 labelme JSON，返回 shapes 列表（每项含 label, points, shape_type）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shapes", [])


def _points_to_int(pts):
    """将 [[x,y],...] 转为 (N,2) int32 数组。"""
    return np.array(pts, dtype=np.float32).astype(np.int32)


def draw_shapes_on_image(img_bgr, shapes, label_colors=None):
    """
    在 BGR 图上绘制 labelme 的 shapes（多边形、矩形等），返回绘制后的图。
    label_colors: { "label_name": (B, G, R) }，缺省则按类别自动分配颜色。
    """
    if label_colors is None:
        label_colors = {}
    palette = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 165, 0),
    ]
    labels_seen = []
    out = img_bgr.copy()
    for sh in shapes:
        label = sh.get("label", "")
        pts = sh.get("points", [])
        shape_type = (sh.get("shape_type") or "polygon").lower()
        if not pts:
            continue
        if label not in label_colors:
            if label not in labels_seen:
                labels_seen.append(label)
            label_colors[label] = palette[labels_seen.index(label) % len(palette)]
        color = label_colors[label]
        if shape_type == "rectangle" and len(pts) == 2:
            x1, y1 = int(pts[0][0]), int(pts[0][1])
            x2, y2 = int(pts[1][0]), int(pts[1][1])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            pts_int = _points_to_int(pts)
            if len(pts_int) < 2:
                continue
            cv2.polylines(out, [pts_int], isClosed=True, color=color, thickness=2)
            x0, y0 = int(pts_int[0][0]), int(pts_int[0][1])
            cv2.putText(out, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def run_interactive(dir_path: Path):
    """交互式预览：键盘左右或 n/p 切换上一张/下一张，q 退出。"""
    pairs = find_image_json_pairs(dir_path)
    if not pairs:
        print(f"[WARN] 在 {dir_path} 下未找到「图片+同名 json」的对子")
        return
    print(f"[INFO] 共 {len(pairs)} 对 图片+json，用 左/右 或 n/p 切换，q 关闭窗口退出")
    if plt is None:
        print("[ERROR] 需要 matplotlib 进行交互预览: pip install matplotlib")
        return
    if cv2 is None:
        print("[ERROR] 需要 opencv: pip install opencv-python")
        return

    idx = [0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    def show_current():
        ax.clear()
        img_path, json_path = pairs[idx[0]]
        img = cv2.imread(str(img_path))
        if img is None:
            from PIL import Image
            img = np.array(Image.open(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        shapes = load_labelme_shapes(json_path)
        out = draw_shapes_on_image(img, shapes)
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_path.name}  [{idx[0]+1}/{len(pairs)}]")
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        if event.key in ("left", "p"):
            idx[0] = max(0, idx[0] - 1)
            show_current()
        elif event.key in ("right", "n", " "):
            idx[0] = min(len(pairs) - 1, idx[0] + 1)
            show_current()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_current()
    plt.show()


def run_save_previews(dir_path: Path, save_dir: Path):
    """将每张图+标注叠加后保存到 save_dir。"""
    if cv2 is None:
        print("[ERROR] 需要 opencv: pip install opencv-python")
        return
    pairs = find_image_json_pairs(dir_path)
    if not pairs:
        print(f"[WARN] 在 {dir_path} 下未找到「图片+同名 json」的对子")
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for img_path, json_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            try:
                from PIL import Image
                img = np.array(Image.open(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f"[WARN] 无法读取 {img_path}: {e}")
                continue
        shapes = load_labelme_shapes(json_path)
        out = draw_shapes_on_image(img, shapes)
        out_path = save_dir / f"{img_path.stem}_preview{img_path.suffix}"
        cv2.imwrite(str(out_path), out)
    print(f"[OK] 已保存 {len(pairs)} 张预览图到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="预览图片与 labelme JSON 标注（支持同目录或 dataset/images + dataset/labels 结构）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dir",
        nargs="?",
        type=Path,
        default=None,
        metavar="DIR",
        help="dataset 根目录：同目录时即图+json 所在目录；或含 images+labels 子目录的根目录",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="dir_opt",
        type=Path,
        default=None,
        help="同上，用选项指定 dataset 目录",
    )
    parser.add_argument(
        "--save", "-s",
        type=Path,
        default=None,
        help="若指定，则将预览图保存到此目录而非交互显示",
    )
    args = parser.parse_args()
    dir_path = args.dir or args.dir_opt
    if dir_path is None:
        parser.error("请指定目录：位置参数 DIR 或 --dir / -d")
    dir_path = Path(dir_path)

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[ERROR] 目录不存在或不是目录: {dir_path}")
        sys.exit(1)

    if args.save is not None:
        run_save_previews(dir_path, Path(args.save))
    else:
        run_interactive(dir_path)


if __name__ == "__main__":
    main()
