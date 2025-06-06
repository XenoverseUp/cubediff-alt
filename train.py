import os
import argparse
import json
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from accelerate import Accelerator

from model import CubeDiff
from config import CubeDiffConfig
from dataset import CubeDiffDataset


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CubeDiff model")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    # Model arguments
    parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Path to pretrained model (e.g., Stable Diffusion)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--cube_size", type=int, default=512, help="Size of cube faces")
    parser.add_argument("--cube_faces", type=int, default=6, help="Number of cube faces")
    parser.add_argument("--fov", type=float, default=95.0, help="Field of view in degrees")
    parser.add_argument("--overlap", type=float, default=2.5, help="Overlap between faces")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_interval", type=int, default=3, help="Evaluate every N epochs")

    # Diffusion arguments
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=0.00085, help="Starting beta value")
    parser.add_argument("--beta_end", type=float, default=0.012, help="Ending beta value")
    parser.add_argument("--prediction_type", type=str, default="v",
                       choices=["epsilon", "v", "x0"], help="Prediction type")

    return parser.parse_args()


def setup_training(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir)

    # Save config
    config = CubeDiffConfig(
        cube_size=args.cube_size,
        cube_faces=args.cube_faces,
        fov=args.fov,
        overlap=args.overlap,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        prediction_type=args.prediction_type
    )

    # Only update pretrained_model_path if provided
    if args.pretrained_model_path:
        config.pretrained_model_path = args.pretrained_model_path

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    return config, accelerator, logger


def create_dataloaders(args, accelerator):
    # Create datasets
    train_dataset = CubeDiffDataset(
        data_dir=args.data_dir,
        cube_size=args.cube_size,
        cube_faces=args.cube_faces,
        fov=args.fov,
        overlap=args.overlap,
        split="train"
    )

    eval_dataset = CubeDiffDataset(
        data_dir=args.data_dir,
        cube_size=args.cube_size,
        cube_faces=args.cube_faces,
        fov=args.fov,
        overlap=args.overlap,
        split="val"
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Prepare dataloaders with accelerator
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    return train_dataloader, eval_dataloader


def evaluate(model, eval_dataloader, accelerator, logger, epoch):
    model.eval()
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating (Epoch {epoch})"):
            outputs = model(batch)
            loss = outputs["loss"]

            # Gather loss from all processes
            gathered_losses = accelerator.gather(loss.repeat(1))
            all_losses.extend(gathered_losses.cpu().numpy())

    avg_loss = np.mean(all_losses)

    if accelerator.is_main_process:
        logger.info(f"Epoch {epoch}, Eval Loss: {avg_loss:.4f}")

    model.train()
    return avg_loss


def train(args):
    # Setup training
    config, accelerator, logger = setup_training(args)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create model
    model = CubeDiff(config=config, device=device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(args, accelerator)

    # Create scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Prepare model, optimizer, and scheduler with accelerator
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    # Training loop
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Forward pass
            outputs = model(batch)
            loss = outputs["loss"]

            # Backward pass
            accelerator.backward(loss)

            # Update weights
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Store loss for epoch average
            epoch_losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

        # Calculate and log epoch average loss
        if epoch_losses:
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch} completed, Average Loss: {epoch_avg_loss:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        # Evaluate every N epochs or on the final epoch
        if (epoch % args.eval_interval == 0) or (epoch == args.num_epochs - 1):
            eval_loss = evaluate(model, eval_dataloader, accelerator, logger, epoch)

            # Save best model
            if eval_loss < best_eval_loss and accelerator.is_main_process:
                best_eval_loss = eval_loss
                model_save_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(model_save_path)
                logger.info(f"New best model saved with eval loss: {eval_loss:.4f} at {model_save_path}")

        # Save checkpoint every N epochs
        if (epoch % args.save_interval == 0) and accelerator.is_main_process:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            model.save_pretrained(checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    # Save final model
    if accelerator.is_main_process:
        final_model_path = os.path.join(args.output_dir, "final_model")
        model.save_pretrained(final_model_path)
        logger.info(f"Training completed. Final model saved to {final_model_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
