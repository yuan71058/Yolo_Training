# Vision Model ToolBox

本仓库整理了 YOLO 目标检测、YOLO 实例分割以及标注数据处理相关脚本，主要用于新数据导入、标注辅助、格式转换、训练集划分和模型训练。

## 目录结构

```text
.
├── yolo26seg/
│   ├── labelme_seg_setup.py              # 准备 labelme 分割标注目录
│   ├── labelme_to_yolo_segments.py       # labelme JSON 转 YOLO segments 格式
│   ├── train_yolo12_seg.py               # YOLO 分割模型训练入口
│   ├── manage_annotation_status.py       # 标注进度管理与未标注数据导出
│   ├── preview_labelme_annotations.py    # 预览 labelme 标注
│   └── readme_yolo26seg.md               # 分割流程详细说明
└── yolov8/
    ├── label_mark.py                     # LabelImg 标注辅助
    ├── voc_xml_to_yolo_txt.py            # Pascal VOC XML 转 YOLO TXT
    ├── merge_train_val_82.py             # 按 8:2 划分 train/val
    ├── train_yolov8.py                   # YOLOv8 训练入口
    ├── train_new_segs.py                 # Mask2Former 分割训练脚本
    └── readme_yolov8.md                  # YOLOv8 流程详细说明
```

## 环境依赖

建议使用 Python 3.9+，并按实际使用的流程安装依赖。

```bash
pip install ultralytics labelme labelImg numpy pillow tqdm torch torchvision transformers tensorboard
```

说明：

- 只训练 YOLO 模型时，重点需要 `ultralytics`。
- 使用 LabelImg 标注检测框时，需要 `labelImg`。
- 使用 labelme 标注分割轮廓时，需要 `labelme`。
- 使用 `train_new_segs.py` 时，需要 `torch`、`transformers`、`tensorboard` 等依赖。

## 快速开始

### 1. YOLOv8 目标检测流程

适用于矩形框检测任务，标注格式为 YOLO `.txt` 或 Pascal VOC `.xml`。

1. 将新图片放入 `label_mark.py` 中配置的 `import_images_dir`。
2. 运行 `yolov8/label_mark.py`，复制图片并启动 LabelImg。
3. 如果 LabelImg 直接保存 YOLO `.txt`，可跳过 XML 转换。
4. 如果 LabelImg 保存 Pascal VOC `.xml`，运行 `yolov8/voc_xml_to_yolo_txt.py` 转为 YOLO `.txt`。
5. 运行 `yolov8/merge_train_val_82.py`，按 8:2 生成训练集和验证集。
6. 运行 `yolov8/train_yolov8.py` 开始训练。

详细说明见 `yolov8/readme_yolov8.md` 或 `yolov8/README.md`。

### 2. YOLO 分割流程

适用于实例分割任务，标注工具为 labelme，标注结果为同名 `.json`。

1. 将新图片放入 `labelme_to_yolo_segments.py` 中配置的 `SOURCE_DIR`，或运行 `yolo26seg/labelme_seg_setup.py` 准备标注目录。
2. 使用 labelme 标注 polygon、rectangle、circle 或 mask，并保存同名 `.json`。
3. 运行 `yolo26seg/labelme_to_yolo_segments.py`，转换为 YOLO segments 格式并划分 train/val。
4. 运行 `yolo26seg/train_yolo12_seg.py` 开始训练。

详细说明见 `yolo26seg/readme_yolo26seg.md`。

## 常用脚本说明

| 脚本 | 用途 |
|------|------|
| `yolov8/label_mark.py` | 导入新图片、生成类别文件、启动 LabelImg，并整理检测标注文件。 |
| `yolov8/voc_xml_to_yolo_txt.py` | 将 Pascal VOC XML 标注转换为 YOLO TXT。 |
| `yolov8/merge_train_val_82.py` | 将图片和标注按 8:2 划分为训练集与验证集。 |
| `yolov8/train_yolov8.py` | 使用 Ultralytics YOLOv8 训练目标检测模型。 |
| `yolo26seg/labelme_seg_setup.py` | 准备 labelme 分割标注目录和类别文件。 |
| `yolo26seg/labelme_to_yolo_segments.py` | 将 labelme JSON 转换为 YOLO segmentation 数据集。 |
| `yolo26seg/train_yolo12_seg.py` | 使用 Ultralytics YOLO 训练分割模型。 |
| `yolo26seg/manage_annotation_status.py` | 检查标注完成情况，并按需导出未标注图片。 |
| `yolo26seg/preview_labelme_annotations.py` | 可视化预览 labelme 标注效果。 |

## 配置注意事项

运行脚本前，需要根据本机数据路径修改脚本顶部的配置区，重点检查：

- 数据根目录，如 `dataset_root`、`SOURCE_DIR`、`OUT_ROOT`、`DATA_ROOT`。
- 类别列表，如 `CLASS_NAMES`、`class_names`、`NUM_CLASSES`。
- 输出目录，如 `SAVE_DIR`、`DST_ROOT`。
- 预训练权重路径，如 `PRETRAINED_WEIGHTS`。

类别顺序必须在标注、转换和训练脚本中保持一致。例如：

```text
0=container
1=food
2=pot
3=slices
```

如果修改类别，请同步更新所有相关脚本和类别文件。

## 数据集格式

训练前通常需要整理为以下结构：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

分割流程还会生成 `data.yaml`，训练脚本中的 `DATA_ROOT` 应与转换脚本的 `OUT_ROOT` 保持一致。

## 建议流程

- 新数据先放入独立的 `incoming` 或待标注目录，避免和已处理数据混在一起。
- 标注前确认类别顺序，标注后再做格式转换。
- 训练前检查图片与标注是否一一对应。
- 大批量训练前，先用少量样本跑通完整流程。
- 推送代码前避免提交本地数据集、训练权重、缓存文件和日志文件。

