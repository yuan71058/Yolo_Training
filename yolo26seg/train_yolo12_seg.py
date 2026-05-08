"""
YOLOv12 实例分割（segment）训练脚本。
使用 Ultralytics 官方 yolo12m-seg.pt（mid 级别），首次运行会先通过 Ultralytics 下载权重；在自建 segment 数据集上训练，类别为 container / food / pot / slices。
训练日志与权重保存到 SAVE_DIR，实验名为 yolo12_seg_{日期_时分}，并启用 TensorBoard。
"""
import urllib.request
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
try:
    from ultralytics.utils.downloads import attempt_download_asset
except ImportError:
    attempt_download_asset = None

# GitHub assets 下载 base（当 attempt_download_asset 不可用或未包含该模型时备用）
ASSETS_RELEASE_TAG = "v8.4.0"  # 若 yolo12m-seg 在更新 release 中可改为对应 tag
ASSETS_URL = f"https://github.com/ultralytics/assets/releases/download/{ASSETS_RELEASE_TAG}"

# =============================================================================
# 训练配置（直接修改此处即可）
# =============================================================================

# ----- 数据与路径 -----
# 数据集根目录（与 labelme_to_yolo_segments 输出一致，其下需有 data.yaml、images/train、images/val、labels/train、labels/val）
DATA_ROOT = Path(r"E:/yolo12_runs")
# 类别数量（需与 labels 中 class_id 一致，0~nc-1）
NUM_CLASSES = 4
# 类别名称（顺序对应 0, 1, 2, 3）
CLASS_NAMES = ["container", "food", "pot", "slices"]
# 训练结果保存根目录（权重、日志、TensorBoard 等）
SAVE_DIR = Path(r"E:\train_pot_1105_2\runs\detect")

# ----- 模型 -----
# 预训练权重：使用官方名称时脚本会先尝试下载。当前 assets 中若有 yolo12m-seg 则用其，否则会退回 yolo11m-seg.pt（mid 级别 seg）
# 也可改为本地 .pt 绝对路径则不再下载
PRETRAINED_WEIGHTS = "yolo12m-seg.pt"
# 当上方模型在 GitHub 不存在时使用的备用（v8.4.0 中 yolo12 仅有 detect，无 seg，故备用 yolo11m-seg）
FALLBACK_WEIGHTS = "yolo11m-seg.pt"

# ----- 设备 -----
DEVICE = 0  # 0=第一块 GPU；多卡可用 [0, 1]；-1=自动；"cpu"=CPU

# ----- 训练轮数与 batch -----
# 总训练轮数。续训（RESUME=True）时必须是「目标总轮数」，例如已训完 100 轮想再训 100 轮则填 200
EPOCHS = 300
BATCH_SIZE = 1  # 显存不足可改为 4 或 2
IMGSZ = 1280    # 输入图像尺寸（正方形）

# ----- 优化器与学习率 -----
LR0 = 0.001
LRF = 0.01
OPTIMIZER = "auto"
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937

# ----- 数据增强与正则 -----
MOSAIC = 1.0
MIXUP = 0.0
PATIENCE = 25      # 早停
SAVE_PERIOD = 10   # 每 N epoch 保存一次 checkpoint

# ----- 其他 -----
# 实验名在 main() 中自动生成为 yolo12_seg_{YYYYMMDD_HHMM}
RESUME = True
# 恢复训练时使用的 checkpoint（last.pt）。留空则自动寻找 SAVE_DIR 下最新的 yolo12_seg_* 目录里的 weights/last.pt
RESUME_CKPT = "E:/train_pot_1105_2/runs/detect/yolo12_seg_20260306_1506/weights/last.pt"
WORKERS = 0
AMP = True
SEED = 42
VAL = True
USE_TENSORBOARD = True


def _download_weights_fallback(model_name: str, fallback_name: str = None) -> str:
    """从 ultralytics/assets 的 GitHub release 下载 .pt 权重，返回本地路径。若 404 且指定了 fallback 则尝试备用。"""
    import urllib.error
    cache_dir = Path.home() / ".cache" / "ultralytics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / model_name
    if local_path.exists():
        return str(local_path)
    url = f"{ASSETS_URL}/{model_name}"
    print(f"[INFO] 从 {url} 下载…")
    try:
        urllib.request.urlretrieve(url, local_path)
    except urllib.error.HTTPError as e:
        if e.code == 404 and fallback_name and fallback_name != model_name:
            print(f"[WARN] {model_name} 不在当前 release，改用 {fallback_name}")
            return _download_weights_fallback(fallback_name, fallback_name=None)
        raise FileNotFoundError(
            f"无法下载 {model_name}。可改用 {fallback_name} 或手动下载后设置 PRETRAINED_WEIGHTS。\n  {e}"
        ) from e
    except Exception as e:
        raise FileNotFoundError(f"下载失败: {e}") from e
    return str(local_path)


def get_data_yaml():
    """若 data.yaml 不存在则生成，确保 nc/names 与当前 CLASS_NAMES 一致；返回 yaml 路径。"""
    data_yaml = DATA_ROOT / "data.yaml"
    content = f"""# YOLO segment dataset
path: {DATA_ROOT.resolve().as_posix()}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(content, encoding="utf-8")
    return str(data_yaml)


def _check_dataset_dirs():
    """检查数据集目录是否存在，缺失时抛出明确错误与解决建议。"""
    root = Path(DATA_ROOT)
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [d for d in required if not d.is_dir()]
    if not missing:
        return
    msg = (
        f"数据集目录不完整，以下路径不存在：\n  " + "\n  ".join(str(p) for p in missing)
        + f"\n\n请任选其一：\n"
        f"  1) 在本机先运行 labelme_to_yolo_segments.py，将 OUT_ROOT 设为与 DATA_ROOT 相同（当前 DATA_ROOT={root}），\n"
        f"     再把生成的 images/train、images/val、labels/train、labels/val 拷到本机该目录下；\n"
        f"  2) 在训练脚本中把 DATA_ROOT 改为本机实际的数据集根路径（需包含上述 4 个子目录）。"
    )
    raise FileNotFoundError(msg)


def _get_epoch_from_ckpt(ckpt_path: Path) -> int | None:
    """从 last.pt 等 checkpoint 中读取已训练到的 epoch，读不到返回 None。"""
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            return int(ckpt["epoch"])
    except Exception:
        pass
    return None


def _find_latest_resume_ckpt(save_dir: Path) -> Path | None:
    """自动寻找 SAVE_DIR 下最新的 yolo12_seg_* / weights/last.pt。"""
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        return None
    runs = []
    for d in save_dir.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith("yolo12_seg_"):
            continue
        ckpt = d / "weights" / "last.pt"
        if ckpt.exists():
            runs.append((ckpt.stat().st_mtime, ckpt))
    if not runs:
        return None
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1]


def main():
    data_yaml_path = get_data_yaml()
    _check_dataset_dirs()
    # 实验名：正常训练为 yolo12_seg_{日期_时分}；恢复训练则沿用 checkpoint 所在目录名
    experiment_name = f"yolo12_seg_{datetime.now():%Y%m%d_%H%M}"
    print(f"[INFO] 数据配置: {data_yaml_path}")
    print(f"[INFO] 类别: {CLASS_NAMES}")
    print(f"[INFO] 保存到: {SAVE_DIR}")

    if USE_TENSORBOARD:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"tensorboard": True})
        print(f"[INFO] TensorBoard 已开启。训练开始后执行: tensorboard --logdir={SAVE_DIR}")

    # 选择权重：RESUME=True 时优先使用上次训练的 last.pt；否则使用预训练权重（并按需下载）
    if RESUME:
        ckpt = Path(RESUME_CKPT) if RESUME_CKPT else None
        if ckpt and ckpt.exists():
            resume_ckpt = ckpt
        else:
            resume_ckpt = _find_latest_resume_ckpt(SAVE_DIR)
        if resume_ckpt is None:
            raise FileNotFoundError(
                f"RESUME=True 但未找到可恢复的 last.pt。请设置 RESUME_CKPT 为具体路径，或确认 {SAVE_DIR} 下存在 yolo12_seg_*\\weights\\last.pt"
            )
        experiment_name = resume_ckpt.parent.parent.name
        ckpt_epoch = _get_epoch_from_ckpt(resume_ckpt)
        if ckpt_epoch is not None:
            print(f"[INFO] 该 checkpoint 已训练到 epoch {ckpt_epoch}，本次目标总轮数 EPOCHS={EPOCHS}")
            if EPOCHS <= ckpt_epoch:
                raise ValueError(
                    f"RESUME=True 时 EPOCHS 必须大于 checkpoint 已训轮数（当前 {ckpt_epoch}）。"
                    f" 请将 EPOCHS 设为「目标总轮数」（例如再训 100 轮则填 {ckpt_epoch + 100}），当前 EPOCHS={EPOCHS} 会导致「nothing to resume」并退出。"
                )
        print(f"[INFO] RESUME=True，将从 checkpoint 恢复: {resume_ckpt}")
        print(f"[INFO] 沿用实验目录名: {experiment_name}")
        model = YOLO(str(resume_ckpt))
    else:
        weights = PRETRAINED_WEIGHTS
        if not Path(weights).is_absolute() and not Path(weights).exists():
            print(f"[INFO] 未找到本地权重 {weights}，正在通过 Ultralytics 下载…")
            if attempt_download_asset is not None:
                try:
                    downloaded = attempt_download_asset(weights)
                    if downloaded and Path(downloaded).exists():
                        weights = downloaded
                    else:
                        weights = _download_weights_fallback(weights, fallback_name=FALLBACK_WEIGHTS)
                except Exception:
                    weights = _download_weights_fallback(weights, fallback_name=FALLBACK_WEIGHTS)
            else:
                weights = _download_weights_fallback(weights, fallback_name=FALLBACK_WEIGHTS)
            print(f"[INFO] 权重已下载至: {weights}")
        print(f"[INFO] 本次实验名: {experiment_name}")
        print(f"[INFO] 加载预训练模型: {weights}")
        model = YOLO(weights)

    train_kwargs = {
        "data": data_yaml_path,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "project": str(SAVE_DIR),
        "name": experiment_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": OPTIMIZER,
        "verbose": True,
        "seed": SEED,
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
        "amp": AMP,
        "resume": RESUME,
        "close_mosaic": 10,
    }

    results = model.train(**train_kwargs)
    print(f"[OK] 训练完成，结果目录: {results.save_dir}")


if __name__ == "__main__":
    main()
