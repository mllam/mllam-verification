import yaml

from .argument_parser import parser
from .config import Config
from .verify import verify

if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()

    # Load the config file into a Config object
    with open(args.config, "r", encoding="utf-8") as file:
        config = Config(**yaml.safe_load(file))

    # Run the verification
    verify(config)
