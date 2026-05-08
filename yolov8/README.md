# YOLOv8 + LabelImg 流程说明

## 一、路径与目录结构

脚本里用到的根目录（需在各自脚本的配置区改成你的实际路径）：

| 用途 | 变量/路径 | 说明 |
|------|-----------|------|
| **新图放入目录** | `label_mark.py` 里的 `import_images_dir` | 新一批原始图片放这里，跑 `label_mark.py` 时会复制到 `images/train` |
| **标注期工作目录** | `label_mark.py` 里的 `dataset_root` | 其下会有 `images/train`、`labels/train`、`class_names.txt` |
| **最终训练数据集** | `merge_train_val_82.py` 的 `DST_ROOT`、`train_yolov8.py` 的 `DATA_ROOT` | 需含 `images/train`、`images/val`、`labels/train`、`labels/val` |

**建议**：把“新图”和“已进流程的图”分开，例如：

- 新图放入：`<dataset_root>\incoming\`  
- 在 `label_mark.py` 中设置：`import_images_dir = os.path.join(dataset_root, "incoming")`  
- `images_out_dir` 保持：`dataset_root/images/train`  

这样每次新数据只往 `incoming` 里丢，跑一次 `label_mark.py` 就会复制到 `images/train` 并进入后续流程。

---

## 二、脚本执行顺序

### 方式 A：LabelImg 直接保存 YOLO 格式（.txt）

1. **放图**  
   把新图片放到 `import_images_dir`（例如 `dataset_root\incoming` 或当前配置的 `images\train`）。

2. **`label_mark.py`**  
   - 把 `import_images_dir` 的图片复制到 `images_out_dir`（即 `dataset_root\images\train`）；  
   - 生成/更新 `class_names.txt` 和各类 `classes.txt`；  
   - 若 `launch_labelimg = True`，会启动 LabelImg。  
   **在 LabelImg 中**：  
   - 格式选 **YOLO**（不要选 PascalVOC）；  
   - 默认保存目录设为 `labels_out_dir`（即 `dataset_root\labels\train`）；  
   - Open Dir 选 `images_out_dir`，标完后 Ctrl+S 保存。  
   关闭 LabelImg 后，脚本会把落在图片目录的 .txt 归位到 `labels/train` 并做校验。

3. **`merge_train_val_82.py`**  
   把 `dataset_root` 下的 `images/train`、`labels/train` 按 8:2 划分为 train/val，并复制到 **最终训练数据集**（`DST_ROOT`），得到 `images/train`、`images/val`、`labels/train`、`labels/val`。

4. **`train_yolov8.py`**  
   在 `DATA_ROOT`（与上一步的 `DST_ROOT` 一致）上训练。运行前在脚本里确认 `DATA_ROOT`、`NUM_CLASSES`、`CLASS_NAMES`、`SAVE_DIR`、`PRETRAINED_WEIGHTS` 等。

---

### 方式 B：LabelImg 保存 Pascal VOC（.xml），再转 YOLO

1. **放图**  
   同上，新图放到 `import_images_dir`。

2. **`label_mark.py`**  
   同上：复制图片到 `images/train`、生成 classes、启动 LabelImg。  
   **在 LabelImg 中**：格式选 **PascalVOC**，保存目录可设为 `images_out_dir` 或任意，只要 XML 和图片同名且最后在 `images/train` 下（或把 XML 放到 `images/train`）。

3. **`voc_xml_to_yolo_txt.py`**  
   把 `IMAGES_TRAIN_DIR`（即 `dataset_root\images\train`）下的 .xml 转成 YOLO 的 .txt，写到 `LABELS_TRAIN_DIR`（即 `dataset_root\labels\train`）。  
   脚本内 `CLASS_NAMES` 顺序需与 `label_mark.py` / `class_names.txt` 一致（0=container, 1=food, 2=pot, 3=slices）。

4. **`merge_train_val_82.py`**  
   同上，8:2 划分并复制到最终训练数据集。

5. **`train_yolov8.py`**  
   同上，在 `DATA_ROOT` 上训练。

---

## 三、顺序小结

| 步骤 | 方式 A（LabelImg 存 YOLO .txt） | 方式 B（LabelImg 存 VOC .xml） |
|------|--------------------------------|--------------------------------|
| 1 | 新图放入 `import_images_dir` | 同左 |
| 2 | 运行 `label_mark.py`，用 LabelImg 标完并保存 .txt | 运行 `label_mark.py`，用 LabelImg 标完并保存 .xml |
| 3 | 运行 `merge_train_val_82.py` | 运行 `voc_xml_to_yolo_txt.py` |
| 4 | 运行 `train_yolov8.py` | 运行 `merge_train_val_82.py` |
| 5 | — | 运行 `train_yolov8.py` |

---

## 四、需要改的配置（按你本机路径）

- **label_mark.py**：`dataset_root`、`import_images_dir`、`images_out_dir`、`labels_out_dir`、可选 `class_names_txt` 和 `default_class_names`。  
- **voc_xml_to_yolo_txt.py**：`IMAGES_TRAIN_DIR`、`LABELS_TRAIN_DIR`、`CLASS_NAMES`（与 label_mark 一致）。  
- **merge_train_val_82.py**：`SRC_IMAGES`、`SRC_LABELS`、`DST_ROOT`（以及可选 `TRAIN_RATIO`、`NEW_DATA_PREFIX`）。  
- **train_yolov8.py**：`DATA_ROOT`（与 merge 的 `DST_ROOT` 一致）、`NUM_CLASSES`、`CLASS_NAMES`、`SAVE_DIR`、`PRETRAINED_WEIGHTS` 等。

类别顺序在所有脚本里必须一致：0=container, 1=food, 2=pot, 3=slices（若你改了类别，请同步改各脚本和 `class_names.txt`）。
