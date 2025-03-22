import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--output-path", type=str, default="calib_dataset.npy")
args = parser.parse_args()


def load_image(img_path: str):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

    return image


def create_calib_dataset(args: dict):
    images_list = [
        img_name
        for img_name in os.listdir(args.data_dir)
        if (
            os.path.splitext(img_name)[1] == ".jpg"
            or os.path.splitext(img_name)[1] == ".png"
        )
        and os.path.isfile(os.path.join(args.data_dir, img_name))
    ]

    images = []
    for img_name in tqdm(images_list, desc="Processing images"):
        image = load_image(os.path.join(args.data_dir, img_name))
        images.append(image)

    images = np.array(images)
    np.save(args.output_path, images)

    print(f"Calib dataset created at {args.output_path}")


if __name__ == "__main__":
    create_calib_dataset(args)
