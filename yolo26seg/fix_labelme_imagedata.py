# -*- coding: utf-8 -*-
"""
修复外包返回的 labelme JSON：当 imageData 为 null 时，从 imagePath 读取图片并写入 base64。
这样用 labelme 打开时不会再报 "a bytes-like object is required, not 'NoneType'"。

用法:
  python fix_labelme_imagedata.py F:\\trial_back\\feedback_zm\\output
  python fix_labelme_imagedata.py --dir F:\\path
"""
import argparse
import base64
import json
import sys
from pathlib import Path


def fix_one_json(json_path: Path) -> bool:
    """若 JSON 中 imageData 为 null，则从 imagePath 读图并写入 base64。返回是否修改过。"""
    json_path = Path(json_path)
    if not json_path.is_file() or json_path.suffix.lower() != ".json":
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("imageData") is not None:
        return False

    image_path = data.get("imagePath")
    if not image_path:
        print(f"[WARN] {json_path.name}: 无 imagePath，跳过")
        return False

    # imagePath 可能是 "..\\trial\\xxx.jpg" 等，相对于当前 JSON 所在目录
    resolved = (json_path.parent / image_path).resolve()
    if not resolved.is_file():
        print(f"[WARN] {json_path.name}: 图片不存在 {resolved}，跳过")
        return False

    try:
        with open(resolved, "rb") as f:
            raw = f.read()
        data["imageData"] = base64.b64encode(raw).decode("ascii")
    except Exception as e:
        print(f"[WARN] {json_path.name}: 读取/编码图片失败: {e}")
        return False

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已修复: {json_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="修复 labelme JSON 中 imageData 为 null 导致打开报错的问题",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dir",
        nargs="?",
        type=Path,
        default=None,
        metavar="DIR",
        help="存放 labelme JSON 的目录（会递归子目录中的 .json）",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="dir_opt",
        type=Path,
        default=None,
        help="同上，用选项指定目录",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="不递归子目录，只处理指定目录下的 .json",
    )
    args = parser.parse_args()
    dir_path = args.dir or args.dir_opt
    if dir_path is None:
        parser.error("请指定目录：位置参数 DIR 或 --dir / -d")
    dir_path = Path(dir_path)

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[ERROR] 目录不存在或不是目录: {dir_path}")
        sys.exit(1)

    if args.no_recurse:
        json_files = list(dir_path.glob("*.json"))
    else:
        json_files = list(dir_path.rglob("*.json"))

    if not json_files:
        print(f"[WARN] 在 {dir_path} 下未找到 .json 文件")
        return

    fixed = 0
    for jp in sorted(json_files):
        if fix_one_json(jp):
            fixed += 1

    print(f"\n共处理 {len(json_files)} 个 JSON，修复 {fixed} 个。")


if __name__ == "__main__":
    main()
