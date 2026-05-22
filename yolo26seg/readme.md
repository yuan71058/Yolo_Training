# YOLO 分割（yolo26seg）新加数据并训练路线

本文档说明：**新一批数据从放入 → 用 labelme 做分割标注 → 转 YOLO segments 并划分 → 训练** 的完整流程与脚本顺序。

---

## 一、路径与目录

| 用途 | 配置位置 | 说明 |
|------|----------|------|
| **新图 / 标注源目录** | `labelme_to_yolo_segments.py` → `SOURCE_DIR` | 已标注的「图片 + 同名 .json」所在目录（如 `seg_dataset/train`） |
| **可选：准备标注用图** | `labelme_seg_setup.py` → `source_root`、`seg_dataset_root` | 从 `source_root` 复制图片到 `seg_dataset_root/train`（及 val），并生成 labelme 用的 `labels.txt` |
| **转换输出 / 训练数据根目录** | `labelme_to_yolo_segments.py` → `OUT_ROOT`<br>`train_yolo12_seg.py` → `DATA_ROOT` | 转换脚本会在此目录下生成 `images/train`、`images/val`、`labels/train`、`labels/val` 及 `data.yaml`；训练脚本的 `DATA_ROOT` 需与此一致 |

**类别顺序**：各脚本中的 `CLASS_NAMES`（如 container, food, pot, slices）必须一致，对应 YOLO 的 class_id 0, 1, 2, 3。

---

## 二、脚本执行顺序

### 标准路线：新图 → labelme 标注 → 转 YOLO → 训练

| 步骤 | 操作 | 脚本/动作 |
|------|------|------------|
| 1 | 准备图片与 labelme 环境 | **方式 1**：把新图片直接放到 `SOURCE_DIR`（如 `<某目录>/seg_dataset/train`）。<br>**方式 2**：运行 **`labelme_seg_setup.py`**，从 `source_root` 复制图片到 `seg_dataset_root/train`（及 val），并生成 `labels.txt`；若 `launch_labelme = True` 会启动 labelme。 |
| 2 | 用 labelme 做分割标注 | 在 labelme 中打开 `SOURCE_DIR`（或 `seg_dataset_root/train`），用 **多边形（polygon）** 或 **矩形（rectangle）** 标出实例轮廓，保存为与图片同名的 **.json**，与图片放在同一目录。支持 polygon、rectangle、circle、mask 等 shape 类型。 |
| 3 | 转 YOLO segments 并划分 train/val | 运行 **`labelme_to_yolo_segments.py`**：读取 `SOURCE_DIR` 下的图片与同名 .json，转为 YOLO 分割格式，并按 8:2 划分到 `OUT_ROOT` 的 `images/train`、`images/val`、`labels/train`、`labels/val`（及生成 `data.yaml`）。 |
| 4 | 训练 | 运行 **`train_yolo12_seg.py`**：在 `DATA_ROOT`（与 `OUT_ROOT` 一致）上训练实例分割模型。确认脚本内 `DATA_ROOT`、`NUM_CLASSES`、`CLASS_NAMES`、`SAVE_DIR`、`PRETRAINED_WEIGHTS` 等配置正确。 |

---

## 三、顺序小结表

| 步骤 | 操作 |
|------|------|
| 1 | 新图放入 `SOURCE_DIR`（或通过 `labelme_seg_setup.py` 复制到 seg_dataset） |
| 2 | 用 **labelme** 标注多边形/轮廓，保存同名 .json |
| 3 | 运行 **`labelme_to_yolo_segments.py`**（转 YOLO seg + 8:2 划分） |
| 4 | 运行 **`train_yolo12_seg.py`**（训练） |

---

## 四、辅助脚本（按需使用）

| 脚本 | 用途 |
|------|------|
| **manage_annotation_status.py** | 检查哪些图片已有/未有 labelme .json；可导出「未标注」图片到外包目录，并维护 completed.txt、outsourced.txt。用法示例：`python manage_annotation_status.py E:\seg_dataset --out-dir E:\outsource_batch --count 100` |
| **preview_labelme_annotations.py** | 预览 labelme 标注结果。 |
| **fix_labelme_imagedata.py** | 修复 labelme JSON 中的 imageData 等字段。 |
| **labelme_reset_geometry.py** | 重置 labelme 中的几何信息。 |
| **labelme_normal.py** | labelme 常规整理。 |
| **compare_labelme_vs_yolo.py** | 对比 labelme 与转换后的 YOLO 标注是否一致。 |

---

## 五、需要修改的配置（按本机路径）

- **labelme_seg_setup.py**（若用）：`source_root`、`seg_dataset_root`、`class_names`，以及 `launch_labelme`、`labelme_cmd`。
- **labelme_to_yolo_segments.py**：`SOURCE_DIR`（图片+同名 .json 所在目录）、`OUT_ROOT`（转换输出根目录）、`CLASS_NAMES`（与 labelme 中 label 一致）。
- **train_yolo12_seg.py**：`DATA_ROOT`（与 `OUT_ROOT` 一致）、`NUM_CLASSES`、`CLASS_NAMES`、`SAVE_DIR`、`PRETRAINED_WEIGHTS`、`RESUME`、`RESUME_CKPT` 等。

**类别顺序**在所有脚本中必须一致（如 0=container, 1=food, 2=pot, 3=slices）。
