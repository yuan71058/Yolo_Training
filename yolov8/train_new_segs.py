import os

# 减轻 TensorFlow / oneDNN 在导入 transformers 时刷 STDERR（与训练逻辑无关）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import math
import time
import random
import zipfile
import hashlib
import argparse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


ADE20K_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE20K_SHA1 = "219e1696abb36c8ba3a3afe7fb2f4b4606a897c7"

def sanitize_ade20k_semantic_mask(
    mask: np.ndarray,
    ignore_index: int,
    max_raw_label_inclusive: int | None,
) -> np.ndarray:
    """
    Mask2Former + do_reduce_labels：仅把像素 0 当作 void；255 会变成 254 仍参与训练，导致 class id 越界。
    官方 ADE20K 语义标注为 0=void、1..150=类；将 255 与非法 id 统一视为 void(0)。
    """
    m = np.asarray(mask, dtype=np.int64).copy()
    m[m == ignore_index] = 0
    m[m < 0] = 0
    if max_raw_label_inclusive is not None and max_raw_label_inclusive > 0:
        m[m > max_raw_label_inclusive] = 0
    return m


class Mask2FormerCollator:
    def __init__(
        self,
        processor,
        ignore_index: int,
        reduce_labels: bool,
        max_raw_label_inclusive: int | None = 150,
    ):
        self.processor = processor
        self.ignore_index = ignore_index
        self.reduce_labels = reduce_labels
        self.max_raw_label_inclusive = max_raw_label_inclusive

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        raw_masks = [item["mask"] for item in batch]
        seg_maps = [
            sanitize_ade20k_semantic_mask(m, self.ignore_index, self.max_raw_label_inclusive)
            for m in raw_masks
        ]
        target_sizes = [tuple(m.shape[-2:]) for m in seg_maps]

        encoded = self.processor(
            images=images,
            segmentation_maps=seg_maps,
            return_tensors="pt",
            do_resize=False,
            ignore_index=self.ignore_index,
            do_reduce_labels=self.reduce_labels,
            size_divisor=32,
        )

        encoded["target_sizes"] = target_sizes
        encoded["gt_seg_maps"] = [torch.from_numpy(m.astype(np.int64)) for m in seg_maps]
        encoded["image_paths"] = [item["image_path"] for item in batch]
        return encoded

def seed_everything(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sha1_of_file(file_path, chunk_size=1024 * 1024):
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, dst_path: str):
    ensure_dir(os.path.dirname(dst_path))

    class _ProgressBar:
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if self.pbar is None:
                self.pbar = tqdm(total=total_size if total_size > 0 else None, unit="B", unit_scale=True, desc="Downloading ADE20K")
            downloaded = block_num * block_size
            if total_size > 0:
                self.pbar.n = min(downloaded, total_size)
            else:
                self.pbar.n = downloaded
            self.pbar.refresh()

    progress = _ProgressBar()
    try:
        urllib.request.urlretrieve(url, dst_path, progress)
    finally:
        if progress.pbar is not None:
            progress.pbar.close()


def extract_zip(zip_path: str, dst_dir: str):
    ensure_dir(dst_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def prepare_ade20k(ade_root: str):
    """
    ade_root 最终会包含：
      ade_root/
        ADEChallengeData2016/
          images/training
          images/validation
          annotations/training
          annotations/validation
    """
    ade_root = os.path.abspath(ade_root)
    ensure_dir(ade_root)

    dataset_dir = os.path.join(ade_root, "ADEChallengeData2016")
    expected_paths = [
        os.path.join(dataset_dir, "images", "training"),
        os.path.join(dataset_dir, "images", "validation"),
        os.path.join(dataset_dir, "annotations", "training"),
        os.path.join(dataset_dir, "annotations", "validation"),
    ]

    if all(os.path.isdir(p) for p in expected_paths):
        print(f"[INFO] ADE20K already prepared: {dataset_dir}")
        return dataset_dir

    zip_path = os.path.join(ade_root, "ADEChallengeData2016.zip")

    need_download = True
    if os.path.isfile(zip_path):
        print(f"[INFO] Found existing zip: {zip_path}")
        try:
            file_sha1 = sha1_of_file(zip_path)
            if file_sha1.lower() == ADE20K_SHA1.lower():
                need_download = False
                print("[INFO] Existing zip SHA1 verified.")
            else:
                print(f"[WARN] SHA1 mismatch, re-downloading. got={file_sha1}")
        except Exception as e:
            print(f"[WARN] SHA1 check failed, re-downloading. reason={e}")

    if need_download:
        print(f"[INFO] Downloading ADE20K to: {zip_path}")
        download_with_progress(ADE20K_URL, zip_path)

        file_sha1 = sha1_of_file(zip_path)
        if file_sha1.lower() != ADE20K_SHA1.lower():
            raise RuntimeError(
                f"ADE20K zip SHA1 mismatch.\nExpected: {ADE20K_SHA1}\nGot     : {file_sha1}"
            )
        print("[INFO] Download complete and SHA1 verified.")

    print("[INFO] Extracting ADE20K zip...")
    extract_zip(zip_path, ade_root)

    if not all(os.path.isdir(p) for p in expected_paths):
        raise RuntimeError(
            "ADE20K extracted, but expected folders are missing:\n" +
            "\n".join(expected_paths)
        )

    print(f"[INFO] ADE20K prepared successfully: {dataset_dir}")
    return dataset_dir


def list_image_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}])


def pair_images_and_masks(images_dir: Path, masks_dir: Path):
    image_files = list_image_files(images_dir)
    mask_map = {p.stem: p for p in masks_dir.iterdir() if p.is_file()}
    pairs = []
    for img_path in image_files:
        stem = img_path.stem
        if stem not in mask_map:
            raise FileNotFoundError(f"找不到与图像同名的 mask: {img_path.name}")
        pairs.append((img_path, mask_map[stem]))
    if not pairs:
        raise RuntimeError(f"没有在 {images_dir} 中找到可用图像")
    return pairs


def resize_keep_aspect(img: Image.Image, mask: Image.Image, long_side: int):  # 保持长边不变，按比例缩放
    w, h = img.size
    scale = long_side / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), resample=Image.BILINEAR)
    mask = mask.resize((nw, nh), resample=Image.NEAREST)
    return img, mask


def pad_to_min_size(img: Image.Image, mask: Image.Image, min_h: int, min_w: int, ignore_index: int):   # 填充到最小尺寸
    w, h = img.size
    pad_w = max(0, min_w - w)
    pad_h = max(0, min_h - h)
    if pad_w > 0 or pad_h > 0:
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=ignore_index)
    return img, mask


def random_crop(img: Image.Image, mask: Image.Image, crop_h: int, crop_w: int):  # 随机裁剪
    w, h = img.size
    if w == crop_w and h == crop_h:
        return img, mask
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    img = img.crop((x, y, x + crop_w, y + crop_h))
    mask = mask.crop((x, y, x + crop_w, y + crop_h))
    return img, mask


class TrainTransform:   # 训练时数据增强
    def __init__(self, image_size: int, ignore_index: int, scale_min=0.5, scale_max=2.0, hflip_prob=0.5):
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hflip_prob = hflip_prob

    def __call__(self, img: Image.Image, mask: Image.Image):
        long_side = int(round(self.image_size * random.uniform(self.scale_min, self.scale_max)))
        img, mask = resize_keep_aspect(img, mask, long_side)

        if random.random() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img, mask = pad_to_min_size(img, mask, self.image_size, self.image_size, self.ignore_index)
        img, mask = random_crop(img, mask, self.image_size, self.image_size)
        return img, mask


class ValTransform:
    def __init__(self, image_size: int, ignore_index: int):
        self.image_size = image_size
        self.ignore_index = ignore_index

    def __call__(self, img: Image.Image, mask: Image.Image):
        img, mask = resize_keep_aspect(img, mask, self.image_size)
        img, mask = pad_to_min_size(img, mask, self.image_size, self.image_size, self.ignore_index)
        return img, mask


class ADE20KDataset(Dataset):  # 数据集类
    def __init__(self, ade_root, split="training", transform=None):
        assert split in {"training", "validation"}
        images_dir = Path(ade_root) / "images" / split
        masks_dir = Path(ade_root) / "annotations" / split

        self.pairs = pair_images_and_masks(images_dir, masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        mask_np = np.array(mask, dtype=np.int64)
        return {
            "image": image,
            "mask": mask_np,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }

'''
def build_collate_fn(processor, ignore_index: int, reduce_labels: bool):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        seg_maps = [item["mask"] for item in batch]
        target_sizes = [tuple(m.shape[-2:]) for m in seg_maps]

        encoded = processor(
            images=images,
            segmentation_maps=seg_maps,
            return_tensors="pt",
            do_resize=False,
            ignore_index=ignore_index,
            do_reduce_labels=reduce_labels,
            size_divisor=32,
        )

        encoded["target_sizes"] = target_sizes
        encoded["gt_seg_maps"] = [torch.from_numpy(m.astype(np.int64)) for m in seg_maps]
        encoded["image_paths"] = [item["image_path"] for item in batch]
        return encoded

    return collate_fn
'''

def move_batch_to_device(batch, device):
    batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
    if "pixel_mask" in batch and batch["pixel_mask"] is not None:
        batch["pixel_mask"] = batch["pixel_mask"].to(device, non_blocking=True)
    if "mask_labels" in batch:
        batch["mask_labels"] = [x.to(device, non_blocking=True) for x in batch["mask_labels"]]
    if "class_labels" in batch:
        batch["class_labels"] = [x.to(device, non_blocking=True) for x in batch["class_labels"]]
    return batch


def fast_confusion_matrix(pred, target, num_classes, ignore_index=255):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)

    inds = num_classes * target + pred
    cm = torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.to(torch.float64)


def compute_miou(confmat: torch.Tensor):
    diag = torch.diag(confmat)
    denom = confmat.sum(1) + confmat.sum(0) - diag
    valid = denom > 0
    iou = torch.zeros_like(diag, dtype=torch.float64)
    iou[valid] = diag[valid] / denom[valid]
    miou = iou[valid].mean().item() if valid.any() else 0.0
    return miou, iou.cpu().numpy()


def colorize_mask(mask: np.ndarray, num_classes: int, ignore_index: int = 255):
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        colors[i] = np.array([(37 * i) % 255, (67 * i) % 255, (97 * i) % 255], dtype=np.uint8)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < num_classes)
    out[valid] = colors[mask[valid]]
    out[mask == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def save_checkpoint(save_path, model, optimizer, scaler, epoch, global_step, best_miou, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_miou": best_miou,
        "args": vars(args),
    }
    torch.save(ckpt, save_path)


def load_checkpoint(load_path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(load_path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def parse_args():
    parser = argparse.ArgumentParser("Train Mask2Former on ADE20K (Windows, auto-download, TensorBoard)")

    parser.add_argument("--ade-root", type=str, default=r"E:\PlayGround\data\ade20k")
    parser.add_argument("--model-name", type=str, default="facebook/mask2former-swin-large-ade-semantic")
    parser.add_argument("--num-classes", type=int, default=150)

    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--reduce-labels", action="store_true", default=True)

    parser.add_argument("--amp", action="store_true", help="开启 mixed precision；默认关闭以提高显存占用")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=r"E:\PlayGround\checkpoints\mask2former_swinl_ade20k")
    parser.add_argument("--logdir", type=str, default=r"E:\PlayGround\runs\detect\mask2former_swinl_ade20k")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    ensure_dir(args.save_dir)
    ensure_dir(args.logdir)

    with open(os.path.join(args.save_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    if torch.cuda.is_available():
        print(f"[INFO] gpu    = {torch.cuda.get_device_name(0)}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_dir = prepare_ade20k(args.ade_root)

    train_ds = ADE20KDataset(
        dataset_dir,
        split="training",
        transform=TrainTransform(args.image_size, args.ignore_index),
    )
    val_ds = ADE20KDataset(
        dataset_dir,
        split="validation",
        transform=ValTransform(args.image_size, args.ignore_index),
    )

    print(f"[INFO] train samples = {len(train_ds)}")
    print(f"[INFO] val samples   = {len(val_ds)}")

    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=False)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(args.model_name)

    id2label = {i: f"class_{i}" for i in range(args.num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    collator = Mask2FormerCollator(
        processor=processor,
        ignore_index=args.ignore_index,
        reduce_labels=args.reduce_labels,
        max_raw_label_inclusive=args.num_classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(2, args.batch_size)),
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers // 2) > 0,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = max(100, int(0.02 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and torch.cuda.is_available())
    writer = SummaryWriter(log_dir=args.logdir, flush_secs=20)

    start_epoch = 0
    global_step = 0
    best_miou = -1.0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"[INFO] resumed from {args.resume}, start_epoch={start_epoch}, best_miou={best_miou:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train {epoch + 1}/{args.epochs}", ncols=120)   # 训练进度条
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in pbar:   # 训练循环
            batch = move_batch_to_device(batch, device)

            with torch.amp.autocast("cuda", enabled=args.amp and torch.cuda.is_available()):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch.get("pixel_mask", None),
                    mask_labels=batch["mask_labels"],
                    class_labels=batch["class_labels"],
                )
                loss = outputs.loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            loss_item = loss.item() * args.grad_accum_steps
            epoch_loss += loss_item
            running_loss += loss_item

            if global_step > 0 and global_step % args.log_interval == 0:
                writer.add_scalar("train/loss_step", running_loss / args.log_interval, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                if torch.cuda.is_available():
                    writer.add_scalar("train/gpu_mem_gb_allocated", torch.cuda.memory_allocated() / 1024**3, global_step)
                    writer.add_scalar("train/gpu_mem_gb_reserved", torch.cuda.memory_reserved() / 1024**3, global_step)
                running_loss = 0.0

            postfix = {
                "loss": f"{loss_item:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if torch.cuda.is_available():
                postfix["mem(GB)"] = f"{torch.cuda.memory_reserved() / 1024**3:.1f}"
            pbar.set_postfix(postfix)

        train_loss_epoch = epoch_loss / max(1, len(train_loader))
        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)

        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            writer.add_scalar("train/gpu_peak_mem_gb", peak_mem_gb, epoch)
            print(f"[INFO] epoch {epoch + 1} peak memory: {peak_mem_gb:.2f} GB")

        latest_ckpt = os.path.join(args.save_dir, "latest.pt")
        save_checkpoint(latest_ckpt, model, optimizer, scaler, epoch, global_step, best_miou, args)

        if (epoch + 1) % args.val_every != 0:
            continue

        model.eval()
        val_loss_sum = 0.0
        confmat = torch.zeros((args.num_classes, args.num_classes), dtype=torch.float64)
        sample_logged = False

        with torch.no_grad():
            pbar_val = tqdm(val_loader, total=len(val_loader), desc=f"Val   {epoch + 1}/{args.epochs}", ncols=120)
            for batch in pbar_val:
                batch = move_batch_to_device(batch, device)

                with torch.amp.autocast("cuda", enabled=args.amp and torch.cuda.is_available()):
                    outputs = model(
                        pixel_values=batch["pixel_values"],
                        pixel_mask=batch.get("pixel_mask", None),
                        mask_labels=batch["mask_labels"],
                        class_labels=batch["class_labels"],
                    )
                    val_loss = outputs.loss

                val_loss_sum += val_loss.item()

                pred_maps = processor.post_process_semantic_segmentation(
                    outputs,
                    target_sizes=batch["target_sizes"],
                )

                for pred, gt in zip(pred_maps, batch["gt_seg_maps"]):
                    pred_cpu = pred.detach().to("cpu", dtype=torch.int64)
                    gt_cpu = gt.to("cpu", dtype=torch.int64)
                    if args.reduce_labels:
                        gt_cpu = gt_cpu.clone()
                        valid = gt_cpu != args.ignore_index
                        gt_cpu[valid] = gt_cpu[valid] - 1
                        gt_cpu[gt_cpu < 0] = args.ignore_index
                    confmat += fast_confusion_matrix(pred_cpu, gt_cpu, args.num_classes, args.ignore_index)

                if not sample_logged and len(pred_maps) > 0:
                    pred_np = pred_maps[0].detach().cpu().numpy().astype(np.int64)  # 预测结果
                    gt_np = batch["gt_seg_maps"][0].cpu().numpy().astype(np.int64)  # 真实标签
                    if args.reduce_labels:
                        gt_np = gt_np.copy()    # 减少标签
                        valid = gt_np != args.ignore_index  # 有效标签
                        gt_np[valid] = gt_np[valid] - 1
                        gt_np[gt_np < 0] = args.ignore_index  # 无效标签

                    pred_color = colorize_mask(pred_np, args.num_classes, args.ignore_index)  # 预测结果颜色化
                    gt_color = colorize_mask(gt_np, args.num_classes, args.ignore_index)  # 真实标签颜色化

                    writer.add_image("val/pred_mask", pred_color, epoch, dataformats="HWC")  # 预测结果图像
                    writer.add_image("val/gt_mask", gt_color, epoch, dataformats="HWC")  # 真实标签图像
                    sample_logged = True

        val_loss_epoch = val_loss_sum / max(1, len(val_loader))
        miou, per_class_iou = compute_miou(confmat)

        writer.add_scalar("val/loss_epoch", val_loss_epoch, epoch)
        writer.add_scalar("val/mIoU", miou, epoch)

        print(f"[INFO] epoch={epoch + 1} train_loss={train_loss_epoch:.4f} val_loss={val_loss_epoch:.4f} mIoU={miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            best_ckpt = os.path.join(args.save_dir, "best_miou.pt")
            save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, global_step, best_miou, args)
            print(f"[INFO] best checkpoint saved: {best_ckpt}, best_mIoU={best_miou:.4f}")

        iou_json = {f"class_{i}": float(v) for i, v in enumerate(per_class_iou.tolist())}
        with open(os.path.join(args.save_dir, f"val_epoch_{epoch + 1:03d}_iou.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "mIoU": miou,
                    "per_class_iou": iou_json,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    writer.close()
    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()