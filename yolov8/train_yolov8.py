"""
YOLOv8 目标检测训练脚本（GPU）。
所有配置在下方「训练配置」中修改，无需命令行参数。
"""
from pathlib import Path
from pickle import TRUE

from ultralytics import YOLO

# =============================================================================
# 训练配置（直接修改此处即可）
# =============================================================================

# ----- 数据与路径 -----
# 数据集根目录（其下需有 images/train、images/val、labels/train、labels/val）
DATA_ROOT = Path(r"E:\train_pot_1105_2\train_pot_1105_2")
# 类别数量（需与 labels 中 class_id 一致，0~nc-1）
NUM_CLASSES = 4
# 类别名称（顺序对应 0, 1, 2, ...）
CLASS_NAMES = ["container", "food", "pot", "slices"]
# 训练结果保存根目录（权重、日志、曲线等）
SAVE_DIR = Path(r"E:\train_pot_1105_2\runs\detect")

# ----- 模型 -----
# 预训练权重：n=nano, s=small, m=medium, l=large, x=extra-large
# 使用 mid 大小即 yolov8m.pt
PRETRAINED_WEIGHTS = "E:/train_pot_1105_2/runs/detect/exp_train_2_resume/weights/best.pt"

# ----- 设备 -----
# 使用 GPU：0 表示第一块 GPU；多卡可用 [0, 1]；-1 表示自动选最空闲的 GPU；"cpu" 表示 CPU
DEVICE = 0

# ----- 训练轮数与 batch -----
# 训练轮数
EPOCHS = 25
# 每批图片数量（-1 表示自动；IMGSZ 较大时显存占用高，需适当改小如 4、8）
BATCH_SIZE = 4
# 输入图像尺寸（正方形边长）。源图 1280x720 时建议 1024 或 1280 以保留细节；越大显存越高、速度越慢
IMGSZ = 1280

# ----- 优化器与学习率 -----
# 初始学习率
LR0 = 0.001
# 最终学习率 = LR0 * LRF
LRF = 0.005
# 优化器：SGD, Adam, AdamW
OPTIMIZER = "auto"
# 权重衰减（正则化）
WEIGHT_DECAY = 0.0005
# 动量（SGD 时有效）
MOMENTUM = 0.937

# ----- 数据增强与正则 -----
# 是否使用马赛克增强
MOSAIC = 1.0
# 是否使用 mixup 增强
MIXUP = 0.0
# 早停耐心值（连续多少 epoch 无提升则停止，0 表示不早停）
PATIENCE = 15
# 保存周期：每多少 epoch 保存一次 checkpoint（-1 表示仅保存最后一轮）
SAVE_PERIOD = 5

# ----- 其他 -----
# 项目名（会作为 SAVE_DIR 下子目录名）
PROJECT_NAME = "train_pot_train_20260226"
# 实验名（用于区分不同 run，如 train1、train2）
EXPERIMENT_NAME = "exp_train_20260226"
# 是否从上次中断的 checkpoint 恢复（RESUME=True 时请将 PRETRAINED_WEIGHTS 改为 last.pt 路径，如 runs/detect/train/weights/last.pt）
RESUME = False
# 工作线程数（DataLoader 的 num_workers，Windows 建议 0 避免多进程报错）
WORKERS = 0
# 是否使用 AMP 混合精度（通常 True 可加速且省显存）
AMP = True
# 随机种子（便于复现）
SEED = 555
# 是否验证时计算 mAP
VAL = True
# 是否写入 TensorBoard 日志（默认关闭，改为 True 后需重新启动训练才会生成 tfevents）
USE_TENSORBOARD = True


def get_data_yaml():
    """生成并写入 data.yaml，返回 yaml 路径。"""
    data_yaml = DATA_ROOT / "data.yaml"
    content = f"""# 自动生成，请勿手动改类别顺序
path: {DATA_ROOT.as_posix()}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(content, encoding="utf-8")
    return str(data_yaml)


def main():
    data_yaml_path = get_data_yaml()
    print(f"[INFO] 数据配置: {data_yaml_path}")

    # 开启 TensorBoard 日志（Ultralytics 默认关闭，不开启则 TensorBoard 会显示 “No dashboards are active”）
    if USE_TENSORBOARD:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"tensorboard": True})
        print(f"[INFO] TensorBoard 已开启。训练开始后在新终端执行: tensorboard --logdir={SAVE_DIR}")

    # 加载预训练模型（mid = yolov8m）
    model = YOLO(PRETRAINED_WEIGHTS)

    # 训练参数（仅列出在脚本中可调的；其余用 ultralytics 默认）
    train_kwargs = {
        "data": data_yaml_path,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "project": str(SAVE_DIR),
        "name": EXPERIMENT_NAME,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": OPTIMIZER,
        "verbose": True,
        "seed": SEED,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 10,
        "resume": RESUME,
        "amp": AMP,
        "fraction": 1.0,
        "profile": False,
        "freeze": None,
        "patience": PATIENCE,
        "save": True,
        "save_period": SAVE_PERIOD,
        "val": VAL,
        "workers": WORKERS,
        "lr0": LR0,
        "lrf": LRF,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
        "mosaic": MOSAIC,
        "mixup": MIXUP,
    }
    train_kwargs["resume"] = RESUME

    results = model.train(**train_kwargs)
    print(f"[OK] 训练完成，结果目录: {results.save_dir}")


if __name__ == "__main__":
    main()
