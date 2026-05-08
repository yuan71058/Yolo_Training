"""
将 Pascal VOC XML 标注转换为 YOLO 标准格式的 .txt 标签。
输入：F:\\chaos_records_img\\images\\train 下的 .xml
输出：F:\\chaos_records_img\\labels\\train 下同名的 .txt（class_id x_center y_center width height，归一化 0~1）
类别顺序：0=container, 1=food, 2=pot, 3=slices
"""
import xml.etree.ElementTree as ET
from pathlib import Path

# 与 label_mark.py 一致的类别顺序
CLASS_NAMES = ["container", "food", "pot", "slices"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGES_TRAIN_DIR = Path(r"F:\chaos_records_img\images\train")
LABELS_TRAIN_DIR = Path(r"F:\chaos_records_img\labels\train")


def parse_voc_xml(xml_path: Path):
    """解析单个 VOC XML，返回 (img_width, img_height, [(class_id, xc, yc, w, h), ...])，坐标已归一化。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        w = int(size.find("width").text or 1)
        h = int(size.find("height").text or 1)
    else:
        w, h = 1, 1

    rows = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        name = (name_el.text or "").strip()
        if name not in CLASS_TO_ID:
            print(f"[WARN] 未知类别 '{name}' 在 {xml_path.name}，已跳过")
            continue
        class_id = CLASS_TO_ID[name]

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.find("xmin").text or 0)
        ymin = float(bbox.find("ymin").text or 0)
        xmax = float(bbox.find("xmax").text or 0)
        ymax = float(bbox.find("ymax").text or 0)

        if w <= 0 or h <= 0:
            continue
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        # 裁剪到 [0,1]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        rows.append((class_id, xc, yc, bw, bh))

    return w, h, rows


def main():
    IMAGES_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = list(IMAGES_TRAIN_DIR.glob("*.xml"))
    if not xml_files:
        print(f"[INFO] 在 {IMAGES_TRAIN_DIR} 下未找到 .xml 文件")
        return

    converted = 0
    for xml_path in sorted(xml_files):
        try:
            w, h, rows = parse_voc_xml(xml_path)
        except Exception as e:
            print(f"[WARN] 解析失败 {xml_path.name}: {e}")
            continue

        txt_path = LABELS_TRAIN_DIR / (xml_path.stem + ".txt")
        lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in rows]
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1

    print(f"[OK] 已转换 {converted} 个 XML -> YOLO .txt，输出目录: {LABELS_TRAIN_DIR}")


if __name__ == "__main__":
    main()
