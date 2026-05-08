"""
将 F:\\chaos_records_img 的 images/labels 按 8:2 划分为 train/val，
并整合复制到 E:\\train_pot_1105_2\\train_pot_1105_2 下对应的 images/labels 的 train/val。
"""
import shutil
import random
from pathlib import Path

# 源：新标注数据
SRC_IMAGES = Path(r"F:\chaos_records_img\images\train")
SRC_LABELS = Path(r"F:\chaos_records_img\labels\train")

# 目标：已有数据集根目录
DST_ROOT = Path(r"E:\train_pot_1105_2\train_pot_1105_2")
DST_IMAGES_TRAIN = DST_ROOT / "images" / "train"
DST_IMAGES_VAL = DST_ROOT / "images" / "val"
DST_LABELS_TRAIN = DST_ROOT / "labels" / "train"
DST_LABELS_VAL = DST_ROOT / "labels" / "val"

# 划分比例 train : val = 8 : 2
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# 图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 复制时若与目标已有文件同名，是否覆盖（False=保留目标原有文件，用前缀区分新文件）
OVERWRITE_IF_EXISTS = False
NEW_DATA_PREFIX = "chaos_"  # 不覆盖时，新文件加此前缀避免重名


def main():
    random.seed(RANDOM_SEED)

    if not SRC_IMAGES.exists():
        raise FileNotFoundError(f"源图片目录不存在: {SRC_IMAGES}")
    if not SRC_LABELS.exists():
        SRC_LABELS.mkdir(parents=True, exist_ok=True)

    # 收集有对应 label 的图片（stem 一致）
    pairs = []
    for img_path in sorted(SRC_IMAGES.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        lab_path = SRC_LABELS / (img_path.stem + ".txt")
        if not lab_path.exists():
            continue
        pairs.append((img_path, lab_path))

    if not pairs:
        print("[WARN] 未找到任何「图片+标签」配对")
        return

    random.shuffle(pairs)
    n = len(pairs)
    n_train = max(1, round(n * TRAIN_RATIO))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    for d in (DST_IMAGES_TRAIN, DST_IMAGES_VAL, DST_LABELS_TRAIN, DST_LABELS_VAL):
        d.mkdir(parents=True, exist_ok=True)

    def copy_pair(img_path: Path, lab_path: Path, out_img_dir: Path, out_lab_dir: Path):
        stem = img_path.stem
        suffix = img_path.suffix
        out_stem = stem
        if not OVERWRITE_IF_EXISTS:
            while (out_img_dir / (out_stem + suffix)).exists():
                out_stem = NEW_DATA_PREFIX + out_stem
        dst_img = out_img_dir / (out_stem + suffix)
        dst_lab = out_lab_dir / (out_stem + ".txt")
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lab_path, dst_lab)
        return out_stem

    train_count = 0
    for img_path, lab_path in train_pairs:
        copy_pair(img_path, lab_path, DST_IMAGES_TRAIN, DST_LABELS_TRAIN)
        train_count += 1

    val_count = 0
    for img_path, lab_path in val_pairs:
        copy_pair(img_path, lab_path, DST_IMAGES_VAL, DST_LABELS_VAL)
        val_count += 1

    print(f"[OK] 已按 8:2 整合到 {DST_ROOT}")
    print(f"     train: {train_count} 对 (图片+标签) -> {DST_IMAGES_TRAIN} / {DST_LABELS_TRAIN}")
    print(f"     val:   {val_count} 对 (图片+标签) -> {DST_IMAGES_VAL} / {DST_LABELS_VAL}")
    if not OVERWRITE_IF_EXISTS:
        print(f"     新文件已加前缀 '{NEW_DATA_PREFIX}' 避免与已有文件重名")


if __name__ == "__main__":
    main()
