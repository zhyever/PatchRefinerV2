import argparse
import copy
import yaml


class DictConfig:
    """Recursive dot-access dictionary, replacing MMEngine ConfigDict.

    Supports both attribute access (cfg.model.type) and dict-like access (cfg['model']['type']).
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(kwargs)
        for k, v in d.items():
            setattr(self, k, self._wrap(v))

    @staticmethod
    def _wrap(v):
        if isinstance(v, dict):
            return DictConfig(v)
        elif isinstance(v, list):
            return [DictConfig._wrap(i) for i in v]
        return v

    def to_dict(self):
        """Convert back to a plain dict (recursive)."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, DictConfig):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [i.to_dict() if isinstance(i, DictConfig) else i for i in v]
            else:
                result[k] = v
        return result

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, self._wrap(value))

    def __contains__(self, key):
        return hasattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __repr__(self):
        return f"DictConfig({self.to_dict()})"

    def copy(self):
        return DictConfig(copy.deepcopy(self.to_dict()))


def load_config(path):
    """Load a YAML config file and return a DictConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DictConfig(raw)


def merge_cli_options(config, options_dict):
    """Merge --cfg-option key=value overrides into config.

    Keys use dot notation: 'general_dataloader.dataset.rgb_image_dir'
    Values are auto-cast to int/float/bool/list where possible.
    """
    for key, value in options_dict.items():
        keys = key.split('.')
        obj = config
        for k in keys[:-1]:
            if not hasattr(obj, k):
                setattr(obj, k, DictConfig({}))
            obj = getattr(obj, k)
        setattr(obj, keys[-1], _auto_cast(value))


def _auto_cast(value):
    """Try to cast string values to appropriate Python types."""
    if not isinstance(value, str):
        return value
    # Strip quotes
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    # Bool
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'none' or value.lower() == 'null':
        return None
    # Int
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # List (e.g., [1,2,3])
    if value.startswith('[') and value.endswith(']'):
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


class DictAction(argparse.Action):
    """Argparse action to parse key=value pairs into a dict.

    Replacement for mmengine.config.DictAction.
    Usage: parser.add_argument('--cfg-option', nargs='+', action=DictAction)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            if '=' not in kv:
                raise ValueError(f"Invalid option format: {kv}. Expected key=value")
            key, val = kv.split('=', 1)
            options[key] = val
        setattr(namespace, self.dest, options)
