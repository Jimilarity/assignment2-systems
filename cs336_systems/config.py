from pathlib import Path

import yaml

def load_yaml(path):
    with Path(path).open(encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def load_model_config(path, size, overrides):
    sizes = load_yaml(path)
    return {**sizes[size], **overrides}