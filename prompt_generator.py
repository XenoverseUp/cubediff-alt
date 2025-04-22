from typing import List, Optional

import io
import re
from ollama import chat
from ollama import ChatResponse
import argparse
import json
import base64
from pprint import pprint

import torch
from torchvision import transforms
from PIL import Image

class OllamaPromptGenerator:
    def __init__(self,
                multi_model_name: str = "gemma3:12b-it-qat",
                host: Optional[str] = "http://localhost:11434"):
        self.model_name = multi_model_name
        self.vision_model_name = multi_model_name
        self.host = host

    def generate_per_face_prompts(self,
                                prompt: str,
                                num_samples: int = 1) -> List[List[str]]:
        """
        Generate detailed descriptions for the six faces of a cubemap.
        """

        system_prompt = (
            "You are an assistant that creates consistent descriptions for the six faces of a "
            "cubemap panorama. The six faces are: front, right, back, left, up, and down. "
            "Given a panorama description, generate a detailed description for each face "
            "that is consistent with the others and forms a coherent 360-degree scene."
            "You MUST format your response exactly with the labels:\n"
            "Front: [description]\n"
            "Right: [description]\n"
            "Back: [description]\n"
            "Left: [description]\n"
            "Up: [description]\n"
            "Down: [description]"
        )


        user_prompt = f"""
        Panorama description: {prompt}

        Generate detailed descriptions for the six faces of a cubemap.
        Each face should be described with a single paragraph.
        The descriptions should be consistent with each other and form a coherent 360-degree scene.

        Format your response as six separate descriptions with clear labels:
        Front: [description]
        Right: [description]
        Back: [description]
        Left: [description]
        Up: [description]
        Down: [description]
        """


        all_face_prompts = []

        try:
            for _ in range(num_samples):
                response: ChatResponse = chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                )

                text = response["message"]["content"]

                face_prompts = []
                for face in ["Front", "Right", "Back", "Left", "Up", "Down"]:
                    pattern = f"{face}: (.*?)(?=\n\n|$)"
                    match = re.search(pattern, text, re.DOTALL)

                    if match:
                        face_prompts.append(match.group(1).strip())
                    else:
                        face_prompts.append(f"{prompt} ({face.lower()} view)")

                all_face_prompts.append(face_prompts)

        except Exception as e:
            print(f"Error generating per-face prompts with Ollama: {e}")
            all_face_prompts = self.fallback_generate(prompt, num_samples)

        return all_face_prompts

    def fallback_generate(self,
                        prompt: str,
                        num_samples: int = 1) -> List[List[str]]:
        """
        Fallback prompt generation without using Ollama.
        Simply uses directional variations of the main prompt.
        """
        all_face_prompts = []
        for _ in range(num_samples):
            face_prompts = [
                f"{prompt} (front view)",
                f"{prompt} (right view)",
                f"{prompt} (back view)",
                f"{prompt} (left view)",
                f"{prompt} (up view)",
                f"{prompt} (down view)"
            ]
            all_face_prompts.append(face_prompts)

        return all_face_prompts

    def _encode_image_to_base64(self, image) -> str:
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        elif isinstance(image, torch.Tensor):
            if image.dim() == 4 and image.size(0) == 1:  # [1, C, H, W]
                image = image.squeeze(0)

            if image.dim() != 3:
                raise ValueError(f"Expected image tensor of shape (C, H, W) or (1, C, H, W), got {image.shape}")

            image = transforms.ToPILImage()(image)

        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image, file path, or torch tensor")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_image(self,
                     image,
                     prompt: str = "Describe this image in detail. What is shown in this view?") -> str:
        """
        Analyze an image using Ollama's vision model and return a detailed description.
        """
        try:
            image_b64 = self._encode_image_to_base64(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_b64}
                    ]
                }
            ]

            response: ChatResponse = chat(
                model=self.vision_model_name,
                messages=messages
            )

            return response["message"]["content"].strip()

        except Exception as e:
            print(f"Error analyzing image with Ollama: {e}")
            return "A view from a panorama scene."

    def generate_panorama_prompt_from_image(self,
                                          image,
                                          face_idx: int = 0,
                                          user_prompt: Optional[str] = None) -> str:
        """
        Generate a panorama description from a conditional face image.
        Optionally combines with a user-provided prompt if available.
        """

        face_names = ["front", "right", "back", "left", "up", "down"]
        face_name = face_names[face_idx]

        # Analyze the image
        system_prompt = (
            f"This image is the {face_name} view of a 360-degree panorama. "
            f"Based on what you see in this {face_name} view, describe what the entire panorama might look like."
        )

        image_description = self.analyze_image(
            image=image,
            prompt=system_prompt
        )

        # If there's a user prompt, combine it with the image description
        if user_prompt:
            combined_prompt = (
                f"Image description: {image_description}\n\n"
                f"User prompt: {user_prompt}\n\n"
                f"Create a coherent panorama description that incorporates both the image content "
                f"and the user's requested theme or elements."
            )

            try:
                response: ChatResponse = chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that creates detailed panorama descriptions."},
                        {"role": "user", "content": combined_prompt}
                    ]
                )

                final_prompt = response["message"]["content"].strip()

            except Exception as e:
                print(f"Error combining image description with user prompt: {e}")
                final_prompt = f"{image_description} {user_prompt}"
        else:
            final_prompt = image_description

        return final_prompt

    def generate_complete_prompt_set(self,
                                   image=None,
                                   face_idx: int = 0,
                                   user_prompt: Optional[str] = None,
                                   num_samples: int = 1) -> List[List[str]]:
        if image is None and user_prompt is None:
            return self.fallback_generate("A panorama scene", num_samples)

        if image is not None and user_prompt is None:
            panorama_prompt = self.generate_panorama_prompt_from_image(image, face_idx)
        elif image is None and user_prompt is not None:
            panorama_prompt = user_prompt
        else:
            panorama_prompt = self.generate_panorama_prompt_from_image(image, face_idx, user_prompt)

        return self.generate_per_face_prompts(panorama_prompt, num_samples)

def main():
    parser = argparse.ArgumentParser(description="OllamaPromptGenerator Demo")

    # Mode selection
    parser.add_argument("--mode", type=str, default="text2faces",
                        choices=["text2faces", "analyze_image", "image2pano", "complete_set"],
                        help="Operation mode: text2faces, analyze_image, image2pano, or complete_set")

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="Text prompt for panorama description")
    parser.add_argument("--image", type=str, help="Path to input image for image-based modes")
    parser.add_argument("--face_idx", type=int, default=0,
                        help="Face index (0=front, 1=right, 2=back, 3=left, 4=up, 5=down)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of sample sets to generate")

    # Model configuration
    parser.add_argument("--text_model", type=str, default="gemma3:12b-it-qat",
                        help="Ollama model for text generation")
    parser.add_argument("--vision_model", type=str, default="gemma3:12b-it-qat",
                        help="Ollama model for vision analysis")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                        help="Ollama API host")

    # Output options
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output file")

    args = parser.parse_args()

    # Validate required arguments based on mode
    if args.mode == "text2faces" and not args.prompt:
        parser.error("--prompt is required for text2faces mode")

    if args.mode in ["analyze_image", "image2pano", "complete_set"] and not args.image:
        parser.error(f"--image is required for {args.mode} mode")

    generator = OllamaPromptGenerator(
        # model_name=args.text_model,
        multi_model_name=args.vision_model,
        host=args.host
    )

    face_names = ["Front", "Right", "Back", "Left", "Up", "Down"]

    try:
        if args.mode == "text2faces":
            print(f"\nGenerating {args.num_samples} sets of face descriptions for: \"{args.prompt}\"")
            print(f"Using text model: {args.text_model}\n")

            face_prompts = generator.generate_per_face_prompts(
                prompt=args.prompt,
                num_samples=args.num_samples
            )

            for i, sample in enumerate(face_prompts):
                print(f"===== Sample {i+1} =====")
                for j, face_prompt in enumerate(sample):
                    print(f"\n{face_names[j]}:")
                    print(face_prompt)
                print("\n")

            if args.output:
                result = {
                    "mode": "text2faces",
                    "main_prompt": args.prompt,
                    "samples": [
                        {face_names[j].lower(): prompt for j, prompt in enumerate(sample)}
                        for sample in face_prompts
                    ]
                }
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved output to {args.output}")

        elif args.mode == "analyze_image":
            print(f"\nAnalyzing image: {args.image}")
            print(f"Using vision model: {args.vision_model}\n")

            from torchvision import transforms
            from PIL import Image

            image = Image.open(args.image).convert("RGB")
            transform = transforms.ToTensor()
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Analyze image
            analysis = generator.analyze_image(
                image=image_tensor,
                prompt="Describe this image in detail. What is shown in this view?"
            )

            print("\nImage Analysis:")
            print(analysis)

            if args.output:
                result = {
                    "mode": "analyze_image",
                    "image_path": args.image,
                    "analysis": analysis
                }
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved output to {args.output}")

        elif args.mode == "image2pano":
            print(f"\nGenerating panorama description from image: {args.image}")
            if args.prompt:
                print(f"With additional prompt: \"{args.prompt}\"")
            print(f"Using face index: {args.face_idx}")
            print(f"Using vision model: {args.vision_model}")
            print(f"Using text model: {args.text_model}\n")

            image = Image.open(args.image).convert("RGB")
            transform = transforms.ToTensor()
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Generate panorama description
            pano_description = generator.generate_panorama_prompt_from_image(
                image=image_tensor,
                face_idx=args.face_idx,
                user_prompt=args.prompt
            )

            print("\nGenerated Panorama Description:")
            print(pano_description)

            if args.output:
                result = {
                    "mode": "image2pano",
                    "image_path": args.image,
                    "face_idx": args.face_idx,
                    "user_prompt": args.prompt,
                    "panorama_description": pano_description
                }
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved output to {args.output}")

        elif args.mode == "complete_set":
            print(f"\nGenerating complete prompt set from image: {args.image}")
            if args.prompt:
                print(f"With additional prompt: \"{args.prompt}\"")
            print(f"Using face index: {args.face_idx}")
            print(f"Using vision model: {args.vision_model}")
            print(f"Using text model: {args.text_model}\n")

            # Load image
            from PIL import Image
            from torchvision import transforms

            image = Image.open(args.image).convert("RGB")
            transform = transforms.ToTensor()
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Generate complete prompt set
            face_prompts = generator.generate_complete_prompt_set(
                image=image_tensor,
                face_idx=args.face_idx,
                user_prompt=args.prompt,
                num_samples=args.num_samples
            )

            for i, sample in enumerate(face_prompts):
                print(f"===== Sample {i+1} =====")
                for j, face_prompt in enumerate(sample):
                    print(f"\n{face_names[j]}:")
                    print(face_prompt)
                print("\n")

            if args.output:
                result = {
                    "mode": "complete_set",
                    "image_path": args.image,
                    "face_idx": args.face_idx,
                    "user_prompt": args.prompt,
                    "samples": [
                        {face_names[j].lower(): prompt for j, prompt in enumerate(sample)}
                        for sample in face_prompts
                    ]
                }
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved output to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        if args.mode == "text2faces":
            print("Trying fallback method...")
            face_prompts = generator.fallback_generate(
                prompt=args.prompt,
                num_samples=args.num_samples
            )
            print("Fallback prompts:")
            pprint(face_prompts)

if __name__ == "__main__":
    main()

# 1. Generate face prompts from text (original functionality)
# python ollama_prompt.py --mode text2faces --prompt "A mountain cabin with a view of snow-capped peaks"

# 2. Analyze an image with the vision model
# python ollama_prompt.py --mode analyze_image --image your_image.jpg --vision_model llava:latest

# 3. Generate a panorama description from an image
# python ollama_prompt.py --mode image2pano --image your_image.jpg --face_idx 0

# 4. Combine image with a text prompt to guide generation
# python ollama_prompt.py --mode image2pano --image your_image.jpg --prompt "Add a dramatic sunset"

# 5. Generate a complete set of faces from image + text
# python ollama_prompt.py --mode complete_set --image your_image.jpg --prompt "Make it nighttime"
