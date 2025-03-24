import argparse

from hailo_sdk_client import ClientRunner

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert ONNX model to Hailo Archive file")
parser.add_argument(
    "--hw-arch",
    type=str,
    choices=["hailo8", "hailo8l", "hailo8r", "hailo10h", "hailo15h", "hailo15m"],
    required=True,
    help="Target Hailo hardware architecture",
)
parser.add_argument(
    "--onnx-path", type=str, required=True, help="Path to the ONNX model"
)
parser.add_argument(
    "--onnx-model-name",
    type=str,
    default="depth-anything-v2",
    help="Name of the ONNX model",
)
parser.add_argument(
    "--output-path", type=str, required=True, help="Path to save the HAR file"
)
args = parser.parse_args()


def onnx_to_har(
    hw_arch: str,
    onnx_path: str,
    onnx_model_name: str,
    output_path: str,
):
    """
    Convert an ONNX model to a Hailo Archive file.
    """
    runner = ClientRunner(hw_arch=hw_arch)
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        onnx_model_name,
        # Input node name of Depth Anything V2 model
        start_node_names=["image"],
        # Output node name of Depth Anything V2 model
        end_node_names=["/Gather_4"],
    )

    # Save the HAR file
    runner.save_har(output_path)
    print(f"Saved HAR to {output_path}")


if __name__ == "__main__":
    onnx_to_har(
        hw_arch=args.hw_arch,
        onnx_path=args.onnx_path,
        onnx_model_name=args.onnx_model_name,
        output_path=args.output_path,
    )
