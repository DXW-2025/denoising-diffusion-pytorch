import argparse, hashlib, json
from pathlib import Path
from torchvision import datasets


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data/cifar10-raw", help="torchvision 下载目录")
    p.add_argument("--out", default="./data/cifar10-images", help="输出 PNG 文件夹")
    p.add_argument("--limit", type=int, default=None, help="只转前 N 张（调试用）")
    args = p.parse_args()

    root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10.url = (
    "https://pypi.tuna.tsinghua.edu.cn/mirrors/pytorch-data/cifar-10-python.tar.gz")
    ds = datasets.CIFAR10(root=str(root), train=True, download=True)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    for i in range(n):
        img, label = ds[i]
        cls = out / str(label)
        cls.mkdir(exist_ok=True)
        img.save(cls / f"{i:05d}.png")

    manifest = {
        "dataset": "CIFAR-10",
        "split": "train",
        "source": datasets.CIFAR10.url,
        "num_samples": n,
        "version": "official-python-release",
        "archive_sha256": None,
    }
    archive = root / "cifar-10-python.tar.gz"
    if archive.exists():
        manifest["archive_sha256"] = sha256(archive)

    (out / "manifests").mkdir(exist_ok=True)
    (out / "manifests" / "dataset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"wrote {n} images -> {out}; manifest -> {out}/manifests/dataset.json")


if __name__ == "__main__":
    main()