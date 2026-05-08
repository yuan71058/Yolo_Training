# -*- coding: utf-8 -*-
"""
遍历指定路径下的图片，检查是否都有对应的 labelme JSON；
- 已有 json 的图片名写入 completed.txt
- 随机抽取指定数量「没有 json 且不在 outsourced.txt 中」的图片复制到指定路径（用于外包），并将本次新增的图片名追加到 outsourced.txt
- 若目标路径已有 completed.txt 与 outsourced.txt，则从「未标注且未在 outsourced.txt 中」的图片里再挑一批（最多 100 张）；若不足 100 张则把剩余可外包的图片全部复制并全部追加到 outsourced.txt。指定保存路径不存在时会自动 mkdir。
支持：同目录 或 dataset/images + dataset/labels（含 train/val 子目录）结构。
用法:
  python manage_annotation_status.py E:\\dataset --out-dir E:\\outsource_batch --count 100
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = ("images", "image")
LABEL_DIR_NAMES = ("labels", "label")

# 默认管理文件名（英文）
COMPLETED_FILE = "completed.txt"
OUTSOURCED_FILE = "outsourced.txt"


def _find_image_label_dirs(root: Path):
    """若为 image+label 分目录结构，返回 (images_dir, labels_dir)；否则 (None, None)。"""
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


def _collect_all_image_json_pairs(root: Path):
    """
    收集所有「图片路径, 对应 json 路径」对（json 可能不存在）。
    返回 [(img_path, json_path), ...]，覆盖同目录、根级 images+labels、子目录 train/val 的 images+labels。
    """
    root = Path(root)
    if not root.is_dir():
        return []

    result = []

    # 根目录下 images + labels
    img_dir, lab_dir = _find_image_label_dirs(root)
    if img_dir is not None and lab_dir is not None:
        for fp in sorted(img_dir.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
                continue
            result.append((fp, lab_dir / f"{fp.stem}.json"))
        return result

    # 子目录（train/val 等）内部分别为 images+labels
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        si, sl = _find_image_label_dirs(sub)
        if si is not None and sl is not None:
            for fp in sorted(si.iterdir()):
                if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
                    continue
                result.append((fp, sl / f"{fp.stem}.json"))
    if result:
        return result

    # 同目录
    for fp in sorted(root.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in IMG_EXTS:
            continue
        result.append((fp, root / f"{fp.stem}.json"))
    return result


def run(
    dataset_root: Path,
    out_images_dir: Path,
    count: int = 100,
    completed_file: Path = None,
    outsourced_file: Path = None,
    seed: int = None,
):
    """
    dataset_root: 数据集根目录（同目录或含 images+labels）
    out_images_dir: 随机选出的「未标注」图片要复制到的目录（用于外包）
    count: 随机选取数量
    completed_file: 已有 json 的图片名列表输出路径，默认 dataset_root / completed.txt
    outsourced_file: 本次选出的外包图片名列表输出路径，默认 dataset_root / outsourced.txt
    seed: 随机种子，便于复现
    """
    dataset_root = Path(dataset_root)
    out_images_dir = Path(out_images_dir)
    if completed_file is None:
        completed_file = dataset_root / COMPLETED_FILE
    if outsourced_file is None:
        outsourced_file = dataset_root / OUTSOURCED_FILE

    pairs = _collect_all_image_json_pairs(dataset_root)
    if not pairs:
        print(f"[WARN] 在 {dataset_root} 下未找到任何图片")
        return

    completed = []   # [(img_path, json_path)]
    pending = []     # [img_path]
    for img_path, json_path in pairs:
        if json_path.exists():
            completed.append((img_path, json_path))
        else:
            pending.append(img_path)

    # 存「已有 json」的图片名（不含路径，便于与 outsourced 一致）
    completed_names = [p[0].name for p in completed]
    completed_file = Path(completed_file)
    completed_file.parent.mkdir(parents=True, exist_ok=True)
    completed_file.write_text("\n".join(sorted(completed_names)) + "\n", encoding="utf-8")
    print(f"[OK] 已有 json 的图片数: {len(completed_names)}，已写入 {completed_file}")

    # 若已有 outsourced.txt，则只从「未标注且未在 outsourced 名单中」的图片里挑选
    outsourced_file = Path(outsourced_file)
    if outsourced_file.exists():
        existing_outsourced = {
            line.strip() for line in outsourced_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        pending = [p for p in pending if p.name not in existing_outsourced]
        if existing_outsourced:
            print(f"[INFO] 已排除 outsourced.txt 中的 {len(existing_outsourced)} 条，剩余可外包候选: {len(pending)}")

    if not pending:
        print("[INFO] 没有可外包的图片（未标注且未在 outsourced.txt 中）")
        return

    # 随机选取：不足 count 时取全部剩余
    if seed is not None:
        random.seed(seed)
    n = min(count, len(pending))
    chosen = random.sample(pending, n)
    chosen_names = [p.name for p in chosen]

    # 复制到指定路径（若目录不存在则创建）
    out_images_dir.mkdir(parents=True, exist_ok=True)
    for img_path in chosen:
        dst = out_images_dir / img_path.name
        shutil.copy2(img_path, dst)
    if n < count:
        print(f"[OK] 可外包不足 {count} 张，已全部选取 {n} 张并复制到 {out_images_dir}")
    else:
        print(f"[OK] 已随机选取 {n} 张未标注图片复制到 {out_images_dir}")

    # 本次选中的名单追加到 outsourced.txt（与已有内容合并去重后写回）
    outsourced_file.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if outsourced_file.exists():
        existing = {line.strip() for line in outsourced_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    existing.update(chosen_names)
    outsourced_file.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
    print(f"[OK] 已外包名单（含本次 {n} 张）已写入 {outsourced_file}，共 {len(existing)} 条")


def main():
    parser = argparse.ArgumentParser(
        description="检查图片与 json 对应关系，生成 completed.txt，并随机选未标注图片复制到指定路径并记录到 outsourced.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        metavar="DATASET",
        help="数据集根目录（同目录或含 images+labels/train/val 等）",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="dataset_opt",
        type=Path,
        default=None,
        help="同上，用选项指定数据集目录",
    )
    parser.add_argument(
        "--out-dir", "-o",
        required=True,
        type=Path,
        help="随机选出的未标注图片要复制到的目录（用于外包）",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="随机选取的未标注图片数量",
    )
    parser.add_argument(
        "--completed-file",
        type=Path,
        default=None,
        help=f"已有 json 的图片名单输出路径，默认 DATASET/{COMPLETED_FILE}",
    )
    parser.add_argument(
        "--outsourced-file",
        type=Path,
        default=None,
        help=f"已外包图片名单输出路径（本次选中的会与文件内已有名单合并去重），默认 DATASET/{OUTSOURCED_FILE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于复现",
    )
    args = parser.parse_args()

    dataset_root = args.dataset or args.dataset_opt
    if dataset_root is None:
        parser.error("请指定数据集目录：位置参数 DATASET 或 --dir / -d")
    dataset_root = Path(dataset_root)

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"[ERROR] 目录不存在或不是目录: {dataset_root}")
        sys.exit(1)

    run(
        dataset_root=dataset_root,
        out_images_dir=args.out_dir,
        count=args.count,
        completed_file=args.completed_file,
        outsourced_file=args.outsourced_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
