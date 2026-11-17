import os

# 必须在导入 huggingface_hub 前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from tqdm import tqdm


def main():
    raw_dir = Path("/root/autodl-tmp/denoising-diffusion-pytorch/data/celeba-parquet")
    out_dir = Path("/root/autodl-tmp/denoising-diffusion-pytorch/data/celeba-train-128")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 下载训练分片；重新运行会复用已下载的文件
    snapshot_download(
        repo_id="flwrlabs/celeba",
        repo_type="dataset",
        allow_patterns=[
            "img_align+identity+attr/train-*.parquet",
            "README.md",
            "LICENSE",
        ],
        local_dir=str(raw_dir),
        max_workers=2,
    )

    files = sorted(
        (raw_dir / "img_align+identity+attr").glob("train-*.parquet")
    )
    if len(files) != 19:
        raise RuntimeError(f"训练分片不完整：找到 {len(files)} 个，预期 19 个")

    count = 0
    with tqdm(total=162770, desc="导出训练图片") as bar:
        for file in files:
            parquet = pq.ParquetFile(file)

            # 分批读取，避免把整个数据集放入内存
            for batch in parquet.iter_batches(
                batch_size=64,
                columns=["image"],
            ):
                for record in batch.to_pylist():
                    target = out_dir / f"{count:06d}.png"

                    if not target.exists():
                        image_bytes = record["image"]["bytes"]
                        if image_bytes is None:
                            raise ValueError(f"图片 {count} 缺少内嵌数据")

                        with Image.open(BytesIO(image_bytes)) as image:
                            # 保持比例缩放并中心裁剪，避免拉伸人脸
                            image = ImageOps.fit(
                                image.convert("RGB"),
                                (128, 128),
                                method=Image.Resampling.LANCZOS,
                            )
                            # 原子替换，避免中断后留下不完整 PNG
                            temp = target.with_suffix(".tmp")
                            image.save(temp, format="PNG", compress_level=1)
                            temp.replace(target)

                    count += 1
                    bar.update(1)

    if count != 162770:
        raise RuntimeError(f"图片数量异常：{count}")

    print(f"完成：{count} 张")
    print(f"训练目录：{out_dir}")


if __name__ == "__main__":
    main()