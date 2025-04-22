import os
import json
import requests
import numpy as np
from pathlib import Path
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import shutil
import random
import math
from projection import CubeProjection

def download_file(url: str, save_path: Path) -> None:
    """Download a file from a given URL."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_polyhaven_hdris(category: str = None, limit: int = math.inf) -> list:
    """Fetch list of HDRIs from Polyhaven."""
    base_url = "https://api.polyhaven.com"
    
    # Get list of all assets
    response = requests.get(f"{base_url}/assets?type=hdris")
    response.raise_for_status()
    assets = response.json()
    
    hdris = []
    i = 0
    for asset_id in tqdm(assets, "Fetching HDRIs..."):
        i += 1
        if i > limit: break
        asset_response = requests.get(f"{base_url}/info/{asset_id}")
        asset_response.raise_for_status()
        asset_info = asset_response.json()
        
        # Get file info
        files_response = requests.get(f"{base_url}/files/{asset_id}")
        files_response.raise_for_status()
        files_info = files_response.json()
        
        if 'hdri' in files_info:
            hdri_info = files_info['hdri']
            if '1k' in hdri_info and 'hdr' in hdri_info['1k']:
                hdr_url = hdri_info['1k']['hdr']['url']
                if category is None or asset_info.get('category') == category:
                    hdris.append({
                        'id': asset_id,
                        'name': asset_info.get('name', ''),
                        'category': asset_info.get('category', ''),
                        'url': hdr_url
                    })
    
    return hdris

def hdr_to_png(hdr_img, exposure=1.0, gamma=2.2):
    """
    Convert HDR image to PNG with tone mapping.
    
    Args:
        hdr_img: HDR image as numpy array
        exposure: Exposure adjustment factor
        gamma: Gamma correction value
    
    Returns:
        PNG-compatible image as numpy array
    """
    # Apply exposure adjustment
    img = hdr_img * exposure
    
    # Reinhard tone mapping
    img = img / (1.0 + img)
    
    # Gamma correction
    img = np.power(img, 1.0 / gamma)
    
    # Clamp to [0, 1] and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img

def process_sample(args: tuple) -> None:
    """Process a single HDRI sample."""
    hdri_info, output_dir, cube_size = args
    
    try:
        # Create directories
        sample_id = hdri_info['id']
        faces_dir = output_dir / "faces" / sample_id
        faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Download equirectangular image
        equirect_path = faces_dir / f"{sample_id}_eqr.hdr"
        
        if not equirect_path.exists():
            download_file(hdri_info['url'], equirect_path)
        
        # Convert to cubemap using OpenCV to read HDR
        equirect_img = cv2.imread(str(equirect_path), cv2.IMREAD_UNCHANGED)
        if equirect_img is None:
            raise ValueError(f"Failed to load HDR image: {equirect_path}")
        
        # Convert from BGR to RGB
        equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
        
        # Save equirectangular image as PNG
        equirect_png_path = faces_dir / "equirect.png"
        equirect_png = hdr_to_png(equirect_img, exposure=1.0, gamma=2.2)
        cv2.imwrite(str(equirect_png_path), cv2.cvtColor(equirect_png, cv2.COLOR_RGB2BGR))
        
        # Convert to PyTorch tensor - HDR images have a wide dynamic range, so we don't normalize
        equirect_tensor = torch.from_numpy(equirect_img).float()
        # Add batch and channel dimensions: [H, W, C] -> [B, C, H, W]
        equirect_tensor = equirect_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Use CubeProjection to convert to cubemap
        cube_proj = CubeProjection()
        cubemap = cube_proj.equirect_to_cubemap(equirect_tensor, cube_size)
        
        # Save cubemap faces - note: face index 4 is actually down and 5 is up in the CubeProjection class
        face_names = ['front', 'right', 'back', 'left', 'down', 'up']  # Swapped 'up' and 'down'
        face_paths = []
        
        # Save both HDR and PNG versions
        for i, face_name in enumerate(face_names):
            # Extract the face and convert to numpy
            face_img = cubemap[0, i].permute(1, 2, 0).numpy()
            
            # Save HDR version
            hdr_path = faces_dir / f"{face_name}.hdr"
            cv2.imwrite(str(hdr_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            # Convert to PNG with tone mapping
            png_path = faces_dir / f"{face_name}.png"
            png_img = hdr_to_png(face_img, exposure=1.0, gamma=2.2)
            cv2.imwrite(str(png_path), cv2.cvtColor(png_img, cv2.COLOR_RGB2BGR))
            
            face_paths.append(png_path.name)  # Use PNG path in metadata
        
        # Create prompt file
        prompts_dir = output_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompts_dir / f"{sample_id}.txt"
        
        prompt = f"A high-quality {hdri_info['category']} environment map showing {hdri_info['name']}"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
            
        return {
            "scene": sample_id,  # Add scene name to metadata
            "faces": face_paths,
            "prompt": prompt_path.name,
            "equirect": equirect_png_path.name
        }
            
    except Exception as e:
        print(f"Error processing {hdri_info['id']}: {str(e)}")
        return None

def prepare_polyhaven_dataset(
    output_dir: str = "polyhaven_dataset",
    cube_size: int = 256,
    category: str = None,
    num_workers: int = 4,
    limit: int = 10,
    train_ratio: float = 0.8
) -> None:
    """Prepare Polyhaven dataset for CubeDiff training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create required directories
    faces_dir = output_dir / "faces"
    prompts_dir = output_dir / "prompts"
    
    faces_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Get HDRIs
    print("Fetching HDRIs from Polyhaven...")
    hdris = get_polyhaven_hdris(category, limit)
    print(f"Found {len(hdris)} HDRIs")
    
    # Process samples
    print("Processing HDRIs...")
    args = [(hdri, output_dir, cube_size) for hdri in hdris]
    metadata = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_sample, args), total=len(args)))
        
    metadata = [result for result in results if result is not None]
    
    random.shuffle(metadata)
    split_idx = int(len(metadata) * train_ratio)
    
    train_metadata = metadata[:split_idx]
    val_metadata = metadata[split_idx:]
    
    # Save metadata
    with open(output_dir / "train_metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(output_dir / "val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    
    print(f"Created dataset with {len(train_metadata)} training samples and {len(val_metadata)} validation samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Polyhaven dataset for CubeDiff training")
    parser.add_argument("--output_dir", type=str, default="polyhaven_dataset", help="Output directory")
    parser.add_argument("--cube_size", type=int, default=256, help="Size of cubemap faces")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (e.g., 'indoor', 'outdoor')")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--limit", type=int, default=math.inf, help="Maximum number of HDRIs to download")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    
    args = parser.parse_args()
    prepare_polyhaven_dataset(**vars(args)) 