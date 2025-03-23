import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Create a calibration dataset for a model")
parser.add_argument(
    "--data-dir", type=str, required=True, help="Path to the data directory"
)
parser.add_argument(
    "--output-path",
    type=str,
    default="calib_dataset.npy",
    help="Path to save the calibration dataset, should end with .npy",
)
parser.add_argument(
    "--image-size",
    type=int,
    nargs=2,
    default=(700, 350),
    help="Image size of the calibration dataset, Should be the same as the input size of the model. (Width, Height)",
)
args = parser.parse_args()


def load_image(img_path: str, image_size: tuple[int, int]):
    image = cv2.imread(img_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to float32
    image = image.astype(np.float32)

    # Calculate ratios for width and height
    width_ratio = image_size[0] / image.shape[1]
    height_ratio = image_size[1] / image.shape[0]

    # Use the larger ratio so both dimensions meet or exceed the target size
    scale = max(width_ratio, height_ratio)
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Crop the image to the desired size from the center
    start_w = (new_width - image_size[0]) // 2
    start_h = (new_height - image_size[1]) // 2
    image = image[start_h : start_h + image_size[1], start_w : start_w + image_size[0]]

    # Skip Normalization because it's done in the model script.

    return image


def create_calib_dataset(data_dir: str, output_path: str, image_size: tuple[int, int]):
    # Get all image files in the data directory
    images_list = [
        img_name
        for img_name in os.listdir(data_dir)
        if (
            os.path.splitext(img_name)[1] == ".jpg"
            or os.path.splitext(img_name)[1] == ".png"
        )
        and os.path.isfile(os.path.join(data_dir, img_name))
    ]

    # Load and process the images
    images = []
    for img_name in tqdm(images_list, desc="Processing images"):
        image = load_image(
            os.path.join(data_dir, img_name),
            image_size,
        )
        images.append(image)

    # Convert the images to a numpy array
    images = np.array(images)

    # Save the calibration dataset
    np.save(output_path, images)

    print(f"Calib dataset created at {output_path}")


if __name__ == "__main__":
    create_calib_dataset(
        data_dir=args.data_dir,
        output_path=args.output_path,
        image_size=args.image_size,
    )
