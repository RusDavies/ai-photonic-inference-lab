"""NumPy Fashion-MNIST baseline with a 784x64 optical-candidate projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from optical_spike.data import load_cifar10, load_fashion_mnist, make_synthetic_dataset


@dataclass(frozen=True)
class BaselineConfig:
    data_dir: Path = Path("data/fashion-mnist")
    dataset: str = "fashion_mnist"
    output_dir: Path = Path("artifacts/spike/baseline")
    hidden_dim: int = 64
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.003
    seed: int = 7
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    synthetic: bool = False


def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-8, 1.0)
    return float(-np.mean(np.log(clipped)))


def init_params(
    input_dim: int,
    hidden_dim: int,
    classes: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        "w_opt": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(
            np.float32
        ),
        "b_opt": np.zeros(hidden_dim, dtype=np.float32),
        "w_head": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, classes)).astype(
            np.float32
        ),
        "b_head": np.zeros(classes, dtype=np.float32),
    }


def forward(
    params: dict[str, np.ndarray], inputs: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    projection = inputs @ params["w_opt"] + params["b_opt"]
    hidden = relu(projection)
    logits = hidden @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {"projection": projection, "hidden": hidden, "logits": logits}


def evaluate(
    params: dict[str, np.ndarray], inputs: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    probs, cache = forward(params, inputs)
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "projection_mean": float(np.mean(cache["projection"])),
        "projection_std": float(np.std(cache["projection"])),
    }


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for name, grad in grads.items():
        moments[name] = beta1 * moments[name] + (1.0 - beta1) * grad
        velocities[name] = beta2 * velocities[name] + (1.0 - beta2) * (grad * grad)
        m_hat = moments[name] / (1.0 - beta1**step)
        v_hat = velocities[name] / (1.0 - beta2**step)
        params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def train_epoch(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    config: BaselineConfig,
    rng: np.random.Generator,
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    start_step: int,
) -> int:
    indices = rng.permutation(train_images.shape[0])
    step = start_step
    classes = params["b_head"].shape[0]
    for start in range(0, train_images.shape[0], config.batch_size):
        batch_idx = indices[start : start + config.batch_size]
        x = train_images[batch_idx]
        y = train_labels[batch_idx]
        y_one_hot = one_hot(y, classes)

        probs, cache = forward(params, x)
        batch_size = x.shape[0]

        d_logits = (probs - y_one_hot) / batch_size
        grad_w_head = cache["hidden"].T @ d_logits
        grad_b_head = np.sum(d_logits, axis=0)
        d_hidden = d_logits @ params["w_head"].T
        d_projection = d_hidden * (cache["projection"] > 0.0)
        grad_w_opt = x.T @ d_projection
        grad_b_opt = np.sum(d_projection, axis=0)

        grads = {
            "w_opt": grad_w_opt.astype(np.float32),
            "b_opt": grad_b_opt.astype(np.float32),
            "w_head": grad_w_head.astype(np.float32),
            "b_head": grad_b_head.astype(np.float32),
        }
        step += 1
        adam_update(params, grads, moments, velocities, step, config.learning_rate)
    return step


def limit_samples(
    images: np.ndarray,
    labels: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None:
        return images, labels
    return images[:limit], labels[:limit]


def run_baseline(config: BaselineConfig) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    if config.synthetic:
        train_images, train_labels, test_images, test_labels = make_synthetic_dataset(
            seed=config.seed
        )
    elif config.dataset == "fashion_mnist":
        train_images, train_labels, test_images, test_labels = load_fashion_mnist(config.data_dir)
    elif config.dataset == "cifar10":
        train_images, train_labels, test_images, test_labels = load_cifar10(config.data_dir)
    else:
        raise ValueError(
            f"Unsupported dataset {config.dataset!r}; choose fashion_mnist, cifar10, or --synthetic"
        )

    train_images, train_labels = limit_samples(train_images, train_labels, config.max_train_samples)
    test_images, test_labels = limit_samples(test_images, test_labels, config.max_test_samples)

    params = init_params(train_images.shape[1], config.hidden_dim, 10, rng)
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    velocities = {name: np.zeros_like(value) for name, value in params.items()}

    history: list[dict[str, float | int]] = []
    step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        step = train_epoch(
            params, train_images, train_labels, config, rng, moments, velocities, step
        )
        train_metrics = evaluate(params, train_images, train_labels)
        test_metrics = evaluate(params, test_images, test_labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    final_train = evaluate(params, train_images, train_labels)
    final_test = evaluate(params, test_images, test_labels)
    elapsed_seconds = time.time() - started

    config.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config.output_dir / "baseline_weights.npz"
    metrics_path = config.output_dir / "baseline_metrics.json"
    np.savez_compressed(weights_path, **params)

    metrics: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dataset": "synthetic" if config.synthetic else config.dataset,
        "train_samples": int(train_images.shape[0]),
        "test_samples": int(test_images.shape[0]),
        "elapsed_seconds": elapsed_seconds,
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "artifacts": {
            "weights": str(weights_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
