import argparse
import os

from hailo_sdk_client import ClientRunner

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--har-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
args = parser.parse_args()


# -----------------------------------------------------------------------------
# Compile the model
# -----------------------------------------------------------------------------
def compile(har_path: str, output_path: str):
    """
    Compile the model and save the HEF file.
    """
    assert os.path.isfile(har_path), f"File {har_path} not found"

    runner = ClientRunner(har=har_path)
    hef = runner.compile()
    with open(output_path, "wb") as f:
        f.write(hef)

    print(f"Saved HEF to {output_path}")


if __name__ == "__main__":
    compile(har_path=args.har_path, output_path=args.output_path)
