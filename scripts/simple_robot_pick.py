#!/usr/bin/env python3
"""
Simple robot pick prediction script using existing robot scene image
"""

import os

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


def predict_robot_pick_action():
    """
    Use the robot scene image to predict pick actions
    """

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")
    print("Loading ReflectVLM diffusion model...")

    # Load model
    model_path = "yunhaif/ReflectVLM-diffusion"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device)

    # Input image path
    input_image_path = "robot_scene_20250731_164731.jpg"

    if not os.path.exists(input_image_path):
        print(f"Error: {input_image_path} not found!")
        return

    # Create output directory
    os.makedirs("robot_pick_predictions", exist_ok=True)

    # Different pick actions to try
    pick_actions = [
        "pick up the yellow object",
        "grasp the bottle",
        "pick up the red can",
        "grab the sugar box",
        "pick up the blue box",
        "grasp the nearest object",
    ]

    # Load input image
    input_image = Image.open(input_image_path).convert("RGB")
    original_size = input_image.size
    input_image_512 = input_image.resize((512, 512))

    print(f"Input image loaded: {original_size}")
    print("Generating predictions...")

    # Generate predictions for each action
    for i, action in enumerate(pick_actions):
        print(f"\n[{i + 1}/{len(pick_actions)}] Generating: '{action}'")

        # Set seed for reproducibility
        generator = torch.Generator(device).manual_seed(42 + i)

        try:
            # Generate prediction
            result = pipe(
                action,
                image=input_image_512,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=10.0,
                generator=generator,
            )

            generated_image = result.images[0]
            generated_image = generated_image.resize(original_size)

            # Save result
            output_path = f"robot_pick_predictions/pick_{i + 1}_{action.replace(' ', '_').replace(',', '')}.png"
            generated_image.save(output_path)
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  Error generating '{action}': {e}")

    print("\nDone! Check 'robot_pick_predictions/' folder for results")


if __name__ == "__main__":
    predict_robot_pick_action()
