#!/usr/bin/env python
"""Train a sparse autoencoder (SAE) on top of frozen DINOv3 features."""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from dinov3.hub.backbones import dinov3_vitl16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default="E:/play_ground/binary/complete_subset",
        help="Root directory that holds image folders.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum directory depth (relative to root) to crawl for images.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=336,
        help="Image size (short and long edge) fed into DINOv3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training the SAE.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of SAE training epochs."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers for image decoding.",
    )
    parser.add_argument(
        "--dinov3-weights",
        type=str,
        default="D:/Explores/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="Path to the locally downloaded DINOv3 ViT-L/16 checkpoint.",
    )
    parser.add_argument(
        "--attach-layer",
        type=str,
        default="last",
        help=(
            "Module path inside the DINOv3 backbone where SAE attaches. "
            "Use 'last' to consume the backbone forward output or provide a dot-separated module path "
            "(e.g. blocks.22.norm1)."
        ),
    )
    parser.add_argument(
        "--attach-pooling",
        type=str,
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy when the tapped feature map has token/spatial dimensions.",
    )
    parser.add_argument(
        "--sae-hidden-dim",
        type=int,
        default=8192,
        help="Hidden width of the sparse autoencoder.",
    )
    parser.add_argument(
        "--sae-activation",
        type=str,
        default="topk",
        choices=["topk", "relu", "gelu", "identity"],
        help="Activation used in the SAE bottleneck.",
    )
    parser.add_argument(
        "--sae-topk",
        type=int,
        default=32,
        help="Number of active units kept by the Top-K activation (ignored for other activations).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for the SAE optimizer.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer used for training the SAE parameters.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay applied by the optimizer.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="How many iterations to wait before logging running metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sae_runs",
        help="Directory for saving checkpoints and run metadata.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Optional epoch interval for intermediate SAE checkpoints (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when shuffling the dataset.",
    )
    return parser.parse_args()


EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def collect_images(root: Path, max_depth: int) -> List[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root {root} was not found.")

    base_depth = len(root.parts)
    images: List[Path] = []

    for dir_path, _, file_names in os.walk(root):
        dir_depth = len(Path(dir_path).parts) - base_depth
        if dir_depth > max_depth:
            continue
        for file_name in file_names:
            lower = file_name.lower()
            if lower.endswith(EXTENSIONS):
                images.append(Path(dir_path) / file_name)

    if not images:
        raise RuntimeError(
            f"No images with extensions {EXTENSIONS} were found under {root} (depth <= {max_depth})."
        )

    images.sort()
    return images


class FolderDataset(Dataset):
    """Simple dataset that returns transformed images from a directory tree."""

    def __init__(self, root: str, max_depth: int, image_size: int) -> None:
        super().__init__()
        self.paths = collect_images(Path(root), max_depth)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.transform(image)


def resolve_module(root_module: nn.Module, dotted_path: str) -> nn.Module:
    current = root_module
    for segment in dotted_path.split("."):
        if not segment:
            continue
        if segment.isdigit():
            current = current[int(segment)]  # type: ignore[index]
        else:
            if not hasattr(current, segment):
                raise AttributeError(f"Module {current.__class__.__name__} has no submodule '{segment}'.")
            current = getattr(current, segment)
    return current


class DINOv3FeatureExtractor(nn.Module):
    """Wraps a DINOv3 backbone and exposes features at an arbitrary layer."""

    def __init__(self, backbone: nn.Module, attach_layer: str = "last", pooling: str = "mean") -> None:
        super().__init__()
        self.backbone = backbone
        self.attach_layer = attach_layer
        self.pooling = pooling
        self._buffer = None
        self._hook = None

        if attach_layer != "last":
            target = resolve_module(self.backbone, attach_layer)
            self._hook = target.register_forward_hook(self._store_activation)

    def _store_activation(self, module, inputs, output):  # pylint: disable=unused-argument
        self._buffer = output

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.attach_layer == "last":
            feats = self.backbone(images)
        else:
            self._buffer = None
            _ = self.backbone(images)
            if self._buffer is None:
                raise RuntimeError(
                    "Forward hook on attach_layer did not capture any features. Check module path."
                )
            feats = self._buffer
        return self._post_process(feats)

    def _post_process(self, feats: torch.Tensor) -> torch.Tensor:
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 4:
            # Feature map from a convolutional stage -> global average pool.
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim == 3:
            # Token embeddings -> CLS or mean pooling.
            if self.pooling == "cls":
                feats = feats[:, 0]
            else:
                feats = feats.mean(dim=1)
        return feats

    def feature_dim(self, sample: torch.Tensor) -> int:
        with torch.no_grad():
            feats = self.forward(sample)
        return feats.shape[-1]

    def cleanup(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class TopKActivation(nn.Module):
    """Retains the top-k activations (by magnitude) for each sample."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.k <= 0 or self.k >= inputs.shape[-1]:
            return inputs
        k = min(self.k, inputs.shape[-1])
        values = torch.topk(inputs.abs(), k, dim=-1).values
        threshold = values[..., -1:].detach()
        mask = inputs.abs() >= threshold
        return inputs * mask


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8192,
        activation: str = "topk",
        topk: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = self._build_activation(activation, topk)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    @staticmethod
    def _build_activation(name: str, topk: int) -> nn.Module:
        name = name.lower()
        if name == "topk":
            return TopKActivation(topk)
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name == "identity":
            return nn.Identity()
        raise ValueError(f"Unknown SAE activation '{name}'.")

    def forward(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:  # type: ignore[override]
        hidden = self.activation(self.encoder(inputs))
        recon = self.decoder(hidden)
        return recon, hidden


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def save_checkpoint(model: nn.Module, output_dir: Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"sae_{tag}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def load_dinov3(weights_path: str) -> nn.Module:
    model = dinov3_vitl16(pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FolderDataset(args.data_root, args.max_depth, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    backbone = load_dinov3(args.dinov3_weights)
    backbone.to(device)
    feature_extractor = DINOv3FeatureExtractor(backbone, args.attach_layer, args.attach_pooling)
    feature_extractor.to(device)

    with torch.no_grad():
        sample = dataset[0].unsqueeze(0).to(device)
        feature_dim = feature_extractor.feature_dim(sample)
    print(f"Feature dimension from DINOv3 at '{args.attach_layer}': {feature_dim}")

    sae = SparseAutoencoder(
        input_dim=feature_dim,
        hidden_dim=args.sae_hidden_dim,
        activation=args.sae_activation,
        topk=args.sae_topk,
    ).to(device)

    optimizer = build_optimizer(args.optimizer, sae.parameters(), args.lr, args.weight_decay)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    total_steps = len(dataloader)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        sae.train()
        running_loss = 0.0
        running_sparsity = 0.0
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
            total=total_steps,
        )
        for step, images in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                features = feature_extractor(images)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                recon, hidden = sae(features)
                loss = criterion(recon, features)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(sae.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            sparsity = (hidden.abs() > 0).float().mean().item()
            running_loss += batch_loss
            running_sparsity += sparsity

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                avg_sparsity = running_sparsity / args.log_every
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "sparsity": f"{avg_sparsity:.3f}"})
                running_loss = 0.0
                running_sparsity = 0.0

        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(sae, output_dir, f"epoch{epoch:03d}")

    save_checkpoint(sae, output_dir, "final")
    feature_extractor.cleanup()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
