import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


def generate_pick_prediction(
    action,
    input_image_path,
    pipe,
    output_path=None,
    num_inference_steps=50,
    image_guidance_scale=1.5,
    guidance_scale=10,
    seed=0,
):
    """
    Generate prediction of robot scene after pick action

    Args:
        action (str): Description of the pick action (e.g., "pick up orange", "grasp the bottle")
        input_image_path (str): Path to input robot scene image
        pipe: Diffusion pipeline model
        output_path (str): Path to save generated image (optional)
        num_inference_steps (int): Number of diffusion steps
        image_guidance_scale (float): Image guidance scale
        guidance_scale (float): Text guidance scale
        seed (int): Random seed for reproducibility

    Returns:
        PIL.Image: Generated image showing scene after pick action
    """

    # Load and preprocess input image
    print(f"Loading input image: {input_image_path}")
    input_image = Image.open(input_image_path).convert("RGB")
    original_size = input_image.size
    print(f"Original image size: {original_size}")

    # Resize to 512x512 for diffusion model
    input_image_resized = input_image.resize((512, 512))

    # Set random seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating prediction for action: '{action}'")
    print(f"Using {num_inference_steps} inference steps...")

    # Generate prediction
    with torch.no_grad():
        result = pipe(
            action,
            image=input_image_resized,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    generated_image = result.images[0]

    # Resize back to original dimensions
    generated_image = generated_image.resize(original_size)

    # Save generated image if output path provided
    if output_path:
        generated_image.save(output_path)
        print(f"Generated image saved to: {output_path}")

    return generated_image


def main():
    parser = argparse.ArgumentParser(
        description="Predict robot scene after pick action"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input robot scene image path"
    )
    parser.add_argument(
        "--action",
        "-a",
        type=str,
        default="pick up object",
        help="Description of pick action to perform",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path for generated image"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yunhaif/ReflectVLM-diffusion",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--img-guidance", type=float, default=1.5, help="Image guidance scale"
    )
    parser.add_argument(
        "--text-guidance", type=float, default=10.0, help="Text guidance scale"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)

    # Generate default output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        output_dir = input_path.parent / "pick_predictions"
        output_dir.mkdir(exist_ok=True)
        args.output = (
            output_dir / f"{input_path.stem}_pick_{args.action.replace(' ', '_')}.png"
        )

    print("=== Robot Pick Action Prediction ===")
    print(f"Input image: {args.input}")
    print(f"Action: {args.action}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
        dtype = torch.float32
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32

    # Load diffusion model
    print("\nLoading diffusion model...")
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.model, torch_dtype=dtype
        ).to(args.device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Generate prediction
    try:
        _generated_image = generate_pick_prediction(
            action=args.action,
            input_image_path=args.input,
            pipe=pipe,
            output_path=args.output,
            num_inference_steps=args.steps,
            image_guidance_scale=args.img_guidance,
            guidance_scale=args.text_guidance,
            seed=args.seed,
        )

        print(f"\nSuccess! Generated prediction for '{args.action}'")
        print(f"Output saved to: {args.output}")

    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
