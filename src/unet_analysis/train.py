import argparse
from pathlib import Path

import mlflow
from torch import optim
from torch.utils.data import DataLoader

from unet_analysis.configs.train_cfg import PairingKeyWords, TrainConfig
from unet_analysis.dataset.dataset import PairedDataset
from unet_analysis.dataset.img_load import load_img
from unet_analysis.dataset.transformation import collate_fn, compose_transformations, random_crop, standardize_img
from unet_analysis.models.simple_unet import UNet
from unet_analysis.utils.trainer import Trainer


def train(train_data_path: Path, train_config: TrainConfig, val_data_path: Path | None = None) -> None:  # noqa: PLR0914
    """Run training process."""
    "Set MLflow experiment."
    mlflow.set_experiment("UNet Analysis")

    "Check image channels."
    image_channels = (
        3
        if train_config.dataset_config is None or train_config.dataset_config.detector is None
        else len(train_config.dataset_config.detector)
    )
    "Set image standardization function."
    mean = (0.485, 0.456, 0.406)[:image_channels]
    std = (0.229, 0.224, 0.225)[:image_channels]
    img_standardization_fn = standardize_img(mean=mean, std=std)

    "Set image augmentation function."
    crop_fn = random_crop(crop_size=train_config.crop_size, seed=train_config.seed)
    augmentation_fn = compose_transformations([crop_fn])

    "Set dataloader."
    val_data_path = val_data_path or train_data_path
    data_loading_fn = load_img(train_config.dataset_config)
    pairing_fn = load_img(train_config.dataset_config, noisy=True)
    train_dataset = PairedDataset(train_data_path, data_loading_fn, img_standardization_fn, pairing_fn, augmentation_fn)
    val_dataset = PairedDataset(val_data_path, data_loading_fn, img_standardization_fn, pairing_fn, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, collate_fn=collate_fn)

    "Create model, optimizer, and loss function for training."
    model = UNet(in_ch=image_channels, out_ch=image_channels).to(train_config.device)
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    "Set trainer for training process."
    trainer = Trainer(train_config, model, optimizer, train_loader, val_loader, train_config.device)

    "Start MLflow run for experiment tracking."
    with mlflow.start_run():
        "Run training loop."
        iteration = 0
        while iteration < train_config.iterations:
            train_loss = trainer.train_step()
            mlflow.log_metrics(train_loss, step=iteration)
            iteration += 1

            "Validation"
            if iteration % train_config.interval == 0 or iteration in {1, train_config.iterations}:
                model.eval()
                val_loss = trainer.val_step()
                mlflow.log_metrics(val_loss, step=iteration)
                model.train()


def main() -> None:
    """Main function to run training process."""
    parser = argparse.ArgumentParser(description="Train UNet model for image denoising.")
    parser.add_argument("--train-data-path", type=Path, required=True, help="Path to training data.")
    parser.add_argument("--val-data-path", type=Path, default=None, help="Path to validation data (optional).")
    parser.add_argument("--clean-keyword", type=str, default=None, help="Keyword to identify clean images.")
    parser.add_argument("--noisy-keyword", type=str, default=None, help="Keyword to identify noisy images.")
    parser.add_argument(
        "--detector-keywords", type=str, nargs="*", default=None, help="List of detector keywords to filter images."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random operations.")
    parser.add_argument("--crop-size", type=int, default=256, help="Size for random cropping of images.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for training.")
    parser.add_argument("--interval", type=int, default=100, help="Interval for validation during training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training.")
    parser.add_argument("--normalize-type", type=str, default="bn", help="Normalization type for model layers.")
    parser.add_argument("--loss-fn", type=str, default="mse", help="Loss function for training.")
    parser.add_argument("--model-arch", type=str, default="unet", help="Model architecture for training.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./results"), help="Output directory for saving model and logs."
    )
    args = parser.parse_args()
    args = vars(args)

    clean_keyword = args.pop("clean_keyword")
    noisy_keyword = args.pop("noisy_keyword")
    detector_keywords = args.pop("detector_keywords")
    dataset_config = (
        PairingKeyWords(clean=clean_keyword, noisy=noisy_keyword, detector=detector_keywords)
        if clean_keyword is not None
        else None
    )

    training_config = TrainConfig.from_optional_kwargs(
        dataset_config=dataset_config,
        seed=args.pop("seed"),
        crop_size=args.pop("crop_size"),
        batch_size=args.pop("batch_size"),
        iterations=args.pop("iterations"),
        interval=args.pop("interval"),
        learning_rate=args.pop("learning_rate"),
        optimizer=args.pop("optimizer"),
        normalize_type=args.pop("normalize_type"),
        loss_fn=args.pop("loss_fn"),
        model_arch=args.pop("model_arch"),
        output_dir=args.pop("output_dir"),
    )

    "Run training process."
    try:
        train(**args, train_config=training_config)
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
