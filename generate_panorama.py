import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import gc

from model import CubeDiff
from config import CubeDiffConfig

def parse_args():
    parser = argparse.ArgumentParser(description="CubeDiff Panorama Generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/generated", help="Output directory")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--input_image", type=str, default=None, help="Optional conditioning image")
    parser.add_argument("--face_idx", type=int, default=0, help="Face index for conditioning (0-5)")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution per face")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU for inference")
    parser.add_argument("--half_precision", action="store_true", help="Use half precision (float16)")
    return parser.parse_args()

def load_image(image_path, resolution):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def save_cubemap_faces(faces, output_dir, prefix="face"):
    os.makedirs(output_dir, exist_ok=True)
    face_names = ["front", "right", "back", "left", "up", "down"]

    for i, face in enumerate(faces):
        face_img = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Convert to uint8 (0-255)
        face_img = (face_img * 255).astype('uint8')

        # Save using PIL
        face_pil = Image.fromarray(face_img)
        face_path = os.path.join(output_dir, f"{prefix}_{face_names[i]}.png")
        face_pil.save(face_path)
        print(f"Saved face to {face_path}")

def save_equirectangular(panorama, output_dir, prefix="panorama"):
    os.makedirs(output_dir, exist_ok=True)

    # Convert to uint8 (0-255)
    pano_img = panorama.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pano_img = (pano_img * 255).astype('uint8')

    # Save using PIL
    pano_pil = Image.fromarray(pano_img)
    pano_path = os.path.join(output_dir, f"{prefix}.png")
    pano_pil.save(pano_path)
    print(f"Saved panorama to {pano_path}")

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Free CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Determine device
    device = torch.device("cpu" if args.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model config if available, otherwise use defaults
    config_path = os.path.join(args.checkpoint, "config.json")
    if os.path.exists(config_path):
        config = CubeDiffConfig()
        config.update_from_file(config_path)
    else:
        config = CubeDiffConfig(
            cube_size=args.resolution,
            fov=95.0,
            overlap=2.5,
            prediction_type="v"
        )

    # Create the model - always initialize on CPU first to avoid memory issues
    print(f"Loading model from {args.checkpoint}")

    # Add method to CubeDiffConfig class to load from file if it doesn't exist
    if not hasattr(CubeDiffConfig, "update_from_file"):
        def update_from_file(self, config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        CubeDiffConfig.update_from_file = update_from_file

    # Initialize the model on CPU first
    model = CubeDiff(config=config, device=torch.device("cpu"))

    print("Loading UNet...")
    unet_path = os.path.join(args.checkpoint, "unet", "pytorch_model.bin")
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    model.unet.load_state_dict(unet_state_dict)
    del unet_state_dict
    gc.collect()

    print("Loading VAE...")
    vae_path = os.path.join(args.checkpoint, "vae", "pytorch_model.bin")
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    model.vae.load_state_dict(vae_state_dict)
    del vae_state_dict
    gc.collect()

    # Convert to half precision if requested
    if args.half_precision:
        model = model.half()

    # Move model to the target device
    model = model.to(device)
    print(f"Model moved to {device}")

    # Set model to evaluation mode
    model.eval()

    # Prepare conditioning image if provided
    cond_image = None
    if args.input_image:
        print(f"Loading conditioning image from {args.input_image}")
        cond_image = load_image(args.input_image, args.resolution)
        cond_image = cond_image.to(device)
        if args.half_precision:
            cond_image = cond_image.half()

    # Prepare prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "A detailed panorama"

    print(f"Using prompt: {prompt}")

    # Generate panorama
    with torch.no_grad():
        print("Generating panorama...")
        try:
            output = model.generate_panorama(
                prompts=[prompt],
                cond_face_idx=args.face_idx if cond_image is not None else None,
                cond_face_image=cond_image,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_steps,
                output_type="equirectangular",
                return_dict=True
            )

            # Save outputs
            prefix = Path(args.input_image).stem if args.input_image else "generated"

            # Save faces
            save_cubemap_faces(output["faces"], args.output_dir, prefix=prefix)

            # Save panorama
            save_equirectangular(output["panorama"], args.output_dir, prefix=prefix)

            print("Generation complete!")

        except Exception as e:
            print(f"Error generating panorama: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
