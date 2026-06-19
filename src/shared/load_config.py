"""Load config.

Example:
    from src.utils.shared.load_config import config
    model_name = config["generator"]["name"]

"""

import yaml

from src.shared.paths import PROJECT_ROOT

config_path = PROJECT_ROOT / "config.yaml"


with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
