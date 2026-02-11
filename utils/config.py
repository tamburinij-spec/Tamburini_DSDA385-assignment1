import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a dictionary.
    Expands user and environment variables in paths.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Optionally expand any path-like entries
    def _expand(v: Any) -> Any:
        if isinstance(v, str) and ("~" in v or "$" in v or "%" in v):
            return os.path.expandvars(os.path.expanduser(v))
        if isinstance(v, dict):
            return {k: _expand(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_expand(val) for val in v]
        return v

    return _expand(cfg)

