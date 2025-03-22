import argparse
import os

import numpy as np
from hailo_sdk_client import ClientRunner

parser = argparse.ArgumentParser()
parser.add_argument("--har-path", type=str, required=True)
parser.add_argument("--calib-dataset-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
args = parser.parse_args()


def optimize(args: dict):
    if not os.path.isfile(args.har_path):
        raise FileNotFoundError(f"File {args.har_path} not found")

    if not os.path.isfile(args.calib_dataset_path):
        raise FileNotFoundError(f"File {args.calib_dataset_path} not found")

    runner = ClientRunner(har=args.har_path)
    scripts = "input_normalization = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])"
    runner.load_model_script(scripts)

    calib_dataset = np.load(args.calib_dataset_path)
    runner.optimize(calib_dataset)

    runner.save_har(args.output_path)


if __name__ == "__main__":
    optimize(args)
