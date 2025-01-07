import argparse
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-c",
    "--config",
    help="Path to the config file",
    type=Path,
    default=Path(__file__).parents[1] / "example.yaml",
)
parser.add_argument(
    "-o",
    "--output_path",
    help="Path to which output is saved, e.g. plot files",
    type=Path,
)
