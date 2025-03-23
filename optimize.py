import argparse
import os

import numpy as np
from hailo_sdk_client import ClientRunner

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--har-path", type=str, required=True, help="Path to the Hailo Archive file"
)
parser.add_argument(
    "--calib-dataset-path",
    type=str,
    required=True,
    help="Path to the calibration dataset",
)
parser.add_argument(
    "--output-path", type=str, required=True, help="Path to save the optimized model"
)
args = parser.parse_args()


def optimize(har_path: str, calib_dataset_path: str, output_path: str):
    """
    Quantize a Hailo Archive file using a calibration dataset.
    """
    assert os.path.isfile(har_path), f"File {har_path} not found"
    assert os.path.isfile(calib_dataset_path), f"File {calib_dataset_path} not found"

    # Load the model archive
    runner = ClientRunner(har=har_path)

    # Define the model script
    scripts = "input_normalization = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])"
    runner.load_model_script(scripts)

    # Load the calibration dataset
    calib_dataset = np.load(calib_dataset_path)
    # Quantize and optimize the model
    runner.optimize(calib_dataset)

    # Save the optimized model
    runner.save_har(output_path)


if __name__ == "__main__":
    optimize(
        har_path=args.har_path,
        calib_dataset_path=args.calib_dataset_path,
        output_path=args.output_path,
    )
