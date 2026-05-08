import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# =========================
# 配置区：只改这里
# =========================
# 新图片数据集根目录（class_names.txt 与此目录同根，便于 labelImg 标注）
dataset_root = r"F:\chaos_records_img"
# 类别文件：与 dataset_root 同目录，顺序对应 YOLO 类别 0,1,2,3
class_names_txt = os.path.join(dataset_root, "class_names.txt")
# 默认 4 类：0=container, 1=food, 2=pot, 3=slices（若 class_names.txt 不存在会自动创建）
default_class_names = ["container", "food", "pot", "slices"]

import_images_dir = r"F:\chaos_records_img\images\train"      # 你新加的原始图片放这里
images_out_dir = os.path.join(dataset_root, "images", "train")   # 训练图片目标目录
labels_out_dir = os.path.join(dataset_root, "labels", "train")   # 训练标签目标目录

launch_labelimg = True                           # 是否自动启动 LabelImg
labelimg_cmd = "labelImg"                        # pip 安装一般就是这个；不行就填绝对路径
# =========================


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def read_class_names(p: Path):
    lines = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    # 去重但保持顺序
    seen = set()
    uniq = []
    for x in lines:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    if not uniq:
        raise ValueError(f"class_names.txt 为空或全是注释：{p}")
    return uniq

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_images(src_dir: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    copied = 0
    skipped = 0
    for fp in sorted(src_dir.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in IMG_EXTS:
            continue
        dst = dst_dir / fp.name
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(fp, dst)
        copied += 1
    return copied, skipped

def write_classes_files(class_names, work_dir: Path, images_dir: Path, labels_dir: Path):
    # labelImg 可能会在保存目录/图片目录找 classes.txt，这里两边都放一份更稳
    ensure_dir(work_dir)
    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    content = "\n".join(class_names) + "\n"
    (work_dir / "predefined_classes.txt").write_text(content, encoding="utf-8")
    (images_dir / "classes.txt").write_text(content, encoding="utf-8")
    (labels_dir / "classes.txt").write_text(content, encoding="utf-8")

    return work_dir / "predefined_classes.txt"

def clamp01(x: float):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def validate_and_fix_yolo_label(txt_path: Path, nc: int, do_clamp=True):
    """
    校验 YOLO label：每行 5 列；class_id 合法；后四个在 0~1
    返回 (ok, changed_lines, warn_count)
    """
    if not txt_path.exists():
        return True, 0, 0

    raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    changed = 0
    warn = 0

    for i, raw in enumerate(raw_lines, start=1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            print(f"[WARN] {txt_path.name}:{i} 列数不是5：{s}")
            warn += 1
            continue

        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            print(f"[WARN] {txt_path.name}:{i} 解析失败：{s}")
            warn += 1
            continue

        if cls < 0 or cls >= nc:
            print(f"[WARN] {txt_path.name}:{i} class_id={cls} 超出范围(0~{nc-1})")
            warn += 1
            # 仍然保留，避免误删；你也可以改成 continue
        old = (xc, yc, w, h)

        if do_clamp:
            xc2, yc2, w2, h2 = map(clamp01, (xc, yc, w, h))
            if (xc2, yc2, w2, h2) != old:
                changed += 1
            xc, yc, w, h = xc2, yc2, w2, h2

        out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    new_text = "\n".join(out_lines) + ("\n" if out_lines else "")
    if new_text != (txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""):
        txt_path.write_text(new_text, encoding="utf-8")

    return True, changed, warn

def sync_labels(images_dir: Path, labels_dir: Path, nc: int):
    """
    把 labelImg 生成的 txt 统一归档到 labels_dir：
    - 若 txt 在 images_dir：移动到 labels_dir
    - 若 txt 已在 labels_dir：只做校验/修正
    - 忽略 classes.txt / predefined_classes.txt
    """
    ensure_dir(labels_dir)

    # 收集所有图片 base name
    image_bases = {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}

    moved = 0
    validated = 0
    fixed = 0
    warns = 0

    # 1) 先处理 images_dir 里的同名 txt（这是最常见的 labelImg 行为）
    for base in sorted(image_bases):
        src_txt = images_dir / f"{base}.txt"
        if src_txt.exists() and src_txt.name not in ("classes.txt", "predefined_classes.txt"):
            dst_txt = labels_dir / src_txt.name
            # 若目标已有，保留最新的那个
            if dst_txt.exists():
                if src_txt.stat().st_mtime > dst_txt.stat().st_mtime:
                    dst_txt.unlink()
                    shutil.move(str(src_txt), str(dst_txt))
                    moved += 1
                else:
                    src_txt.unlink()
            else:
                shutil.move(str(src_txt), str(dst_txt))
                moved += 1

    # 2) 校验 labels_dir 中所有 label
    for txt in sorted(labels_dir.glob("*.txt")):
        if txt.name in ("classes.txt", "predefined_classes.txt"):
            continue
        ok, chg, w = validate_and_fix_yolo_label(txt, nc, do_clamp=True)
        if ok:
            validated += 1
            fixed += chg
            warns += w

    return moved, validated, fixed, warns

def launch_labelimg_ui(images_dir: Path, class_file: Path, work_dir: Path):
    """
    启动 labelImg。注意：不同 fork 对命令行参数支持略有差异。
    官方文档支持：labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE] :contentReference[oaicite:1]{index=1}
    """
    cmd = [labelimg_cmd, str(images_dir), str(class_file)]
    print("[INFO] Launch:", " ".join(cmd))
    print("[INFO] 在 LabelImg 里：")
    print("  1) 点工具栏 'PascalVOC' 按钮切到 'YOLO' 格式")
    print("  2) File -> Change default saved annotation folder (或 Ctrl+R) 选择你的 labels_out_dir")
    print("  3) Open Dir 选择 images_out_dir，开始框 bbox，Ctrl+S 保存")
    print("  4) 关掉 LabelImg 后脚本会继续做标签归档/校验")
    subprocess.run(cmd, cwd=str(work_dir), check=False)

def main():
    class_p = Path(class_names_txt)
    src_dir = Path(import_images_dir)
    img_dir = Path(images_out_dir)
    lab_dir = Path(labels_out_dir)

    # 若 class_names.txt 不存在，在数据集根目录下用默认类别自动创建
    if not class_p.exists():
        ensure_dir(class_p.parent)
        content = "\n".join(default_class_names) + "\n"
        class_p.write_text(content, encoding="utf-8")
        print(f"[INFO] 已创建 {class_p}，类别: {default_class_names}")
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    class_names = read_class_names(class_p)
    nc = len(class_names)
    print(f"[INFO] Loaded {nc} classes from {class_p}")

    ensure_dir(img_dir)
    ensure_dir(lab_dir)

    # 工作目录（放 predefined_classes.txt）
    work_dir = class_p.parent / ".labelimg_work"
    ensure_dir(work_dir)

    predefined = write_classes_files(class_names, work_dir, img_dir, lab_dir)

    copied, skipped = copy_images(src_dir, img_dir)
    print(f"[INFO] Images copied: {copied}, skipped (already exists): {skipped}")

    if launch_labelimg:
        launch_labelimg_ui(img_dir, predefined, work_dir)

    moved, validated, fixed, warns = sync_labels(img_dir, lab_dir, nc)
    print(f"[OK] Labels synced. moved={moved}, validated={validated}, fixed_lines={fixed}, warns={warns}")
    print(f"[INFO] Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
