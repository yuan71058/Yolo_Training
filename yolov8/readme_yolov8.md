# YOLOv8 新加数据并训练路线

本文档说明：**新一批数据从放入 → 标定 → 转换（若用 XML）→ 划分合并 → 训练** 的完整流程与脚本顺序。

---

## 一、路径与目录

| 用途 | 配置位置 | 说明 |
|------|----------|------|
| **新图放入目录** | `label_mark.py` → `import_images_dir` | 新来的原始图片放这里，跑 `label_mark.py` 时会复制到 `images/train` |
| **标注工作根目录** | `label_mark.py` → `dataset_root` | 其下有 `images/train`、`labels/train`、`class_names.txt` |
| **最终训练数据集** | `merge_train_val_82.py` → `DST_ROOT`<br>`train_yolov8.py` → `DATA_ROOT` | 需含 `images/train`、`images/val`、`labels/train`、`labels/val`；两处路径要一致 |

**建议**：新图单独目录，避免和已进流程的图混在一起。

- 新图放入：`<dataset_root>\incoming\`
- 在 `label_mark.py` 中设置：`import_images_dir = os.path.join(dataset_root, "incoming")`
- `images_out_dir` 保持：`dataset_root/images/train`

---

## 二、脚本执行顺序

### 路线 A：LabelImg 直接保存 YOLO 格式（.txt）

| 步骤 | 操作 | 脚本/动作 |
|------|------|------------|
| 1 | 放图 | 把新图片放到 `import_images_dir`（如 `dataset_root\incoming`） |
| 2 | 标定 | 运行 **`label_mark.py`**：复制图到 `images/train`、生成 classes、启动 LabelImg；在 LabelImg 中选 **YOLO** 格式，默认保存目录设为 `labels_out_dir`，标完保存并关闭，脚本会自动把 .txt 归位到 `labels/train` 并校验 |
| 3 | 划分并合并 | 运行 **`merge_train_val_82.py`**：将 `dataset_root` 下 train 按 8:2 划分为 train/val，复制到最终训练目录 `DST_ROOT` |
| 4 | 训练 | 运行 **`train_yolov8.py`**：在 `DATA_ROOT`（与 `DST_ROOT` 一致）上训练 |

**不需要**运行 `voc_xml_to_yolo_txt.py`。

---

### 路线 B：LabelImg 保存 Pascal VOC（.xml），再转 YOLO

| 步骤 | 操作 | 脚本/动作 |
|------|------|------------|
| 1 | 放图 | 把新图片放到 `import_images_dir` |
| 2 | 标定 | 运行 **`label_mark.py`**：同上；在 LabelImg 中选 **PascalVOC**，保存 .xml（与图片同名，且最终在 `images/train` 下或可被 `voc_xml_to_yolo_txt.py` 读到） |
| 3 | XML → YOLO .txt | 运行 **`voc_xml_to_yolo_txt.py`**：将 `IMAGES_TRAIN_DIR` 下的 .xml 转为 `LABELS_TRAIN_DIR` 下的 .txt，类别顺序与 `class_names.txt` 一致 |
| 4 | 划分并合并 | 运行 **`merge_train_val_82.py`**：同上，8:2 划分并复制到 `DST_ROOT` |
| 5 | 训练 | 运行 **`train_yolov8.py`**：在 `DATA_ROOT` 上训练 |

---

## 三、顺序小结表

| 步骤 | 路线 A（LabelImg 存 YOLO .txt） | 路线 B（LabelImg 存 VOC .xml） |
|------|--------------------------------|--------------------------------|
| 1 | 新图放入 `import_images_dir` | 同左 |
| 2 | `label_mark.py` → LabelImg 标完存 .txt | `label_mark.py` → LabelImg 标完存 .xml |
| 3 | `merge_train_val_82.py` | `voc_xml_to_yolo_txt.py` |
| 4 | `train_yolov8.py` | `merge_train_val_82.py` |
| 5 | — | `train_yolov8.py` |

---

## 四、需要修改的配置（按本机路径）

- **label_mark.py**：`dataset_root`、`import_images_dir`、`images_out_dir`、`labels_out_dir`，以及可选 `class_names_txt`、`default_class_names`。
- **voc_xml_to_yolo_txt.py**（路线 B）：`IMAGES_TRAIN_DIR`、`LABELS_TRAIN_DIR`、`CLASS_NAMES`（与 label_mark / class_names.txt 一致）。
- **merge_train_val_82.py**：`SRC_IMAGES`、`SRC_LABELS`、`DST_ROOT`，以及可选 `TRAIN_RATIO`、`NEW_DATA_PREFIX`。
- **train_yolov8.py**：`DATA_ROOT`（与 `DST_ROOT` 一致）、`NUM_CLASSES`、`CLASS_NAMES`、`SAVE_DIR`、`PRETRAINED_WEIGHTS` 等。

**类别顺序**在所有脚本中必须一致（如 0=container, 1=food, 2=pot, 3=slices）；若修改类别，请同步改各脚本和 `class_names.txt`。
