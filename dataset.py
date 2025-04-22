import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm


class CubeDiffDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cube_size: int = 512,
        cube_faces: int = 6,
        fov: float = 90.0,
        overlap: float = 0.1,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Dataset for CubeDiff model training.
        
        Args:
            data_dir: Directory containing the dataset
            cube_size: Size of each cube face
            cube_faces: Number of cube faces (default: 6 for cubemap)
            fov: Field of view in degrees
            overlap: Overlap between faces
            split: Dataset split ('train' or 'val')
            transform: Optional transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.cube_size = cube_size
        self.cube_faces = cube_faces
        self.fov = fov
        self.overlap = overlap
        self.split = split
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(cube_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
        # Validate dataset structure
        self._validate_dataset()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata from JSON file."""
        metadata_path = self.data_dir / f"{self.split}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def _validate_dataset(self):
        """Validate dataset structure and files."""
        required_dirs = ["faces", "prompts"]
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        # Validate that all referenced files exist
        for sample in tqdm(self.metadata, desc="Validating dataset"):
            # Get scene name from the first face path (e.g., "front.png" -> scene directory)
            scene_name = sample.get("scene", None)
            if not scene_name:
                # Try to infer scene name from the first face path
                first_face = sample["faces"][0]
                scene_name = first_face.split("/")[0] if "/" in first_face else None
            
            if not scene_name:
                raise ValueError(f"Could not determine scene name for sample: {sample}")
            
            # Check face images in the scene directory
            for face_name in ["front.png", "right.png", "back.png", "left.png", "down.png", "up.png"]:
                face_path = self.data_dir / "faces" / scene_name / face_name
                if not face_path.exists():
                    raise FileNotFoundError(f"Face image not found: {face_path}")
            
            # Check prompt file
            prompt_path = self.data_dir / "prompts" / sample["prompt"]
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    def _load_faces(self, sample: Dict[str, Any]) -> List[torch.Tensor]:
        """Load and preprocess cube faces for a sample."""
        # Get scene name
        scene_name = sample.get("scene", None)
        if not scene_name:
            first_face = sample["faces"][0]
            scene_name = first_face.split("/")[0] if "/" in first_face else None
        
        if not scene_name:
            raise ValueError(f"Could not determine scene name for sample: {sample}")
        
        # Load faces from scene directory
        faces = []
        face_names = ["front.png", "right.png", "back.png", "left.png", "down.png", "up.png"]
        for face_name in face_names:
            face_path = self.data_dir / "faces" / scene_name / face_name
            face_image = Image.open(face_path).convert("RGB")
            face_tensor = self.transform(face_image)
            faces.append(face_tensor)
        return faces
    
    def _load_prompt(self, sample: Dict[str, Any]) -> str:
        """Load text prompt for a sample."""
        prompt_path = self.data_dir / "prompts" / sample["prompt"]
        with open(prompt_path, "r") as f:
            return f.read().strip()
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample.
        
        Returns:
            Dictionary containing:
                - faces: List of face tensors [C, H, W]
                - text: Text prompt string
        """
        sample = self.metadata[idx]
        
        # Load faces and prompt
        faces = self._load_faces(sample)
        text = self._load_prompt(sample)
        
        return {
            "faces": faces,
            "text": text
        }


def create_dataset_metadata(
    data_dir: str,
    split: str = "train",
    train_ratio: float = 0.8
) -> None:
    """
    Create dataset metadata files for train/val splits.
    
    Args:
        data_dir: Directory containing the dataset
        split: Which split to create ('train', 'val', or 'all')
        train_ratio: Ratio of data to use for training
    """
    data_dir = Path(data_dir)
    
    # Get all face directories
    face_dirs = sorted([d for d in (data_dir / "faces").iterdir() if d.is_dir()])
    prompt_files = sorted([f for f in (data_dir / "prompts").iterdir() if f.suffix == ".txt"])
    
    # Create metadata entries
    metadata = []
    for face_dir, prompt_file in zip(face_dirs, prompt_files):
        # Get all face images for this sample
        face_images = sorted([f.name for f in face_dir.iterdir() if f.suffix in [".jpg", ".png"]])
        
        if len(face_images) != 6:  # Assuming 6 faces per cubemap
            print(f"Warning: {face_dir} has {len(face_images)} faces, expected 6")
            continue
        
        metadata.append({
            "faces": face_images,
            "prompt": prompt_file.name
        })
    
    # Split into train/val if needed
    if split in ["train", "val"]:
        np.random.shuffle(metadata)
        split_idx = int(len(metadata) * train_ratio)
        
        if split == "train":
            metadata = metadata[:split_idx]
        else:  # val
            metadata = metadata[split_idx:]
    
    # Save metadata
    output_path = data_dir / f"{split}_metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {split} metadata with {len(metadata)} samples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CubeDiff dataset metadata")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"],
                       help="Which split to create")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training")
    
    args = parser.parse_args()
    create_dataset_metadata(args.data_dir, args.split, args.train_ratio) 