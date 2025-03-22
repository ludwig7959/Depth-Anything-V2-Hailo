import argparse

from hailo_sdk_client import ClientRunner

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert ONNX model to Hailo Archive file")
parser.add_argument(
    "--hw-arch",
    type=str,
    choices=["hailo8", "hailo8l", "hailo10", "hailo15h", "hailo15m"],
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


def onnx_to_har(args: dict):
    runner = ClientRunner(hw_arch=args.hw_arch)
    hn, npz = runner.translate_onnx_model(
        args.onnx_path,
        args.onnx_model_name,
        start_node_names=["image"],
        end_node_names=["depth"],
        net_input_shapes={"image": [1, 3, 480, 640]},
    )

    runner.save_har(args.output_path)
    print(f"Saved HAR to {args.output_path}")


if __name__ == "__main__":
    onnx_to_har(args)
