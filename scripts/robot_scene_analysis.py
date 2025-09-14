#!/usr/bin/env python3
"""
Analyze robot scene and generate contextual pick predictions
"""

import os

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


def analyze_and_predict():
    """
    Analyze the robot scene image and generate contextual pick predictions
    """

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=== Robot Scene Pick Prediction ===")
    print(f"Device: {device}")

    # Check if robot image exists
    robot_image_path = "robot_scene_20250731_164731.jpg"
    if not os.path.exists(robot_image_path):
        print(f"Robot image not found: {robot_image_path}")
        return

    # Display image info
    robot_image = Image.open(robot_image_path)
    print(f"Robot image loaded: {robot_image.size}")

    print("\nLoading ReflectVLM diffusion model...")

    try:
        # Load model
        model_path = "yunhaif/ReflectVLM-diffusion"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(device)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directory
    output_dir = "robot_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Based on the visible objects in robot scenes, create contextual actions
    contextual_actions = [
        # Generic pick actions
        "pick up the object in front of the robot",
        "grasp the nearest object",
        "pick up the item on the table",
        # Specific to typical robot lab objects
        "pick up the yellow object",
        "grasp the red container",
        "pick up the blue item",
        "grab the box",
        "pick up the bottle",
        # More detailed actions
        "carefully pick up the small object",
        "grasp the cylindrical object",
        "pick up the object with the gripper",
    ]

    # Prepare input
    original_size = robot_image.size
    robot_image_512 = robot_image.resize((512, 512))

    print(f"\nGenerating {len(contextual_actions)} pick predictions...")

    successful_generations = 0

    for i, action in enumerate(contextual_actions):
        print(f"\n[{i + 1:2d}/{len(contextual_actions)}] '{action}'")

        try:
            # Set seed for reproducibility
            generator = torch.Generator(device).manual_seed(100 + i)

            # Generate prediction
            with torch.no_grad():
                result = pipe(
                    action,
                    image=robot_image_512,
                    num_inference_steps=40,  # Slightly faster
                    image_guidance_scale=1.5,
                    guidance_scale=12.0,  # Slightly higher for better adherence to prompt
                    generator=generator,
                )

            generated_image = result.images[0]
            generated_image = generated_image.resize(original_size)

            # Save with clean filename
            clean_action = action.replace(" ", "_").replace(",", "").replace(".", "")
            output_path = f"{output_dir}/robot_pick_{i + 1:02d}_{clean_action[:30]}.png"
            generated_image.save(output_path)

            print(f"     Saved: {output_path}")
            successful_generations += 1

        except Exception as e:
            print(f"     Error: {e}")

    print(
        f"\nCompleted! {successful_generations}/{len(contextual_actions)} successful generations"
    )
    print(f"Results saved in: {output_dir}/")

    # Save original image for comparison
    comparison_path = f"{output_dir}/00_original_robot_scene.jpg"
    robot_image.save(comparison_path)
    print(f"Original image copied to: {comparison_path}")


if __name__ == "__main__":
    analyze_and_predict()
