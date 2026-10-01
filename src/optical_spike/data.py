"""Fashion-MNIST loading utilities."""

from __future__ import annotations

import gzip
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


FASHION_MNIST_BASE_URLS = [
    "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion",
]
FASHION_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"
CIFAR10_TRAIN_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))
CIFAR10_TEST_BATCH = "test_batch"


def download_fashion_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FASHION_MNIST_FILES.values():
        target = data_dir / filename
        if target.exists():
            continue
        errors: list[str] = []
        for base_url in FASHION_MNIST_BASE_URLS:
            url = f"{base_url}/{filename}"
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    target.write_bytes(response.read())
                break
            except OSError as exc:
                errors.append(f"{url}: {exc}")
        else:
            joined = "\n".join(errors)
            raise RuntimeError(f"Could not download {filename} from configured mirrors:\n{joined}")


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.astype(np.int64).reshape(count)


def load_fashion_mnist(data_dir: Path, download: bool = True) -> tuple[np.ndarray, ...]:
    if download:
        download_fashion_mnist(data_dir)

    train_images = read_idx_images(data_dir / FASHION_MNIST_FILES["train_images"])
    train_labels = read_idx_labels(data_dir / FASHION_MNIST_FILES["train_labels"])
    test_images = read_idx_images(data_dir / FASHION_MNIST_FILES["test_images"])
    test_labels = read_idx_labels(data_dir / FASHION_MNIST_FILES["test_labels"])
    return train_images, train_labels, test_images, test_labels


def download_cifar10(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / CIFAR10_DIR
    if extracted_dir.exists():
        return
    archive_path = data_dir / CIFAR10_ARCHIVE
    if not archive_path.exists():
        with urllib.request.urlopen(CIFAR10_URL, timeout=60) as response:
            archive_path.write_bytes(response.read())
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(data_dir, filter="data")


def _read_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    images = payload["data"].astype(np.float32) / 255.0
    labels = np.asarray(payload["labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir: Path, download: bool = True) -> tuple[np.ndarray, ...]:
    if download:
        download_cifar10(data_dir)

    cifar_dir = data_dir / CIFAR10_DIR
    train_batches = [_read_cifar_batch(cifar_dir / batch) for batch in CIFAR10_TRAIN_BATCHES]
    train_images = np.concatenate([images for images, _labels in train_batches], axis=0)
    train_labels = np.concatenate([labels for _images, labels in train_batches], axis=0)
    test_images, test_labels = _read_cifar_batch(cifar_dir / CIFAR10_TEST_BATCH)
    return train_images, train_labels, test_images, test_labels


def make_synthetic_dataset(
    train_samples: int = 256,
    test_samples: int = 64,
    input_dim: int = 784,
    classes: int = 10,
    seed: int = 7,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(0.0, 0.7, size=(classes, input_dim)).astype(np.float32)

    def build(samples: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, classes, size=samples, dtype=np.int64)
        noise = rng.normal(0.0, 0.25, size=(samples, input_dim)).astype(np.float32)
        images = prototypes[labels] + noise
        images = 1.0 / (1.0 + np.exp(-images))
        return images.astype(np.float32), labels

    train_images, train_labels = build(train_samples)
    test_images, test_labels = build(test_samples)
    return train_images, train_labels, test_images, test_labels
