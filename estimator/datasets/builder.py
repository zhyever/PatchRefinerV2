"""Simple dataset factory — no MMEngine registry."""

from estimator.datasets.general_dataset import ImageDataset
from estimator.datasets.u4k_dataset import UnrealStereo4kDataset
from estimator.datasets.cityscapes_dataset import CityScapesDataset
from estimator.utils.config import DictConfig

_DATASET_REGISTRY = {
    'ImageDataset': ImageDataset,
    'UnrealStereo4kDataset': UnrealStereo4kDataset,
    'CityScapesDataset': CityScapesDataset,
}


def build_dataset(cfg):
    """Build a dataset from config.

    Args:
        cfg: A DictConfig or dict with a 'type' key indicating the class.

    Returns:
        Dataset instance.
    """
    if hasattr(cfg, 'to_dict'):
        cfg = cfg.to_dict()
    else:
        cfg = dict(cfg)
    dataset_type = cfg.pop('type')
    # Re-wrap nested dicts as DictConfig so datasets can use attribute access
    for k, v in cfg.items():
        if isinstance(v, dict):
            cfg[k] = DictConfig(v)
    cls = _DATASET_REGISTRY.get(dataset_type)
    if cls is None:
        raise KeyError(f'Unknown dataset type: {dataset_type}. '
                       f'Available: {list(_DATASET_REGISTRY.keys())}')
    return cls(**cfg)
