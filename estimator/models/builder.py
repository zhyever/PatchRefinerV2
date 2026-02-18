"""Simple model factory replacing MMEngine Registry.

All model/block/loss classes are registered in a flat dict and
constructed by build_model(cfg) where cfg is a dict with a 'type' key.
"""

_MODEL_REGISTRY = {}


def register(cls):
    """Register a class in the model registry."""
    _MODEL_REGISTRY[cls.__name__] = cls
    return cls


def build_model(cfg):
    """Build a model/block/loss from a config dict.

    Args:
        cfg: dict (or DictConfig) with a 'type' key and constructor kwargs.

    Returns:
        Instantiated module.
    """
    if hasattr(cfg, 'to_dict'):
        cfg = cfg.to_dict()
    else:
        cfg = dict(cfg)
    model_type = cfg.pop('type')
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise KeyError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}")
    return cls(**cfg)


def _register_all():
    """Lazy import and register all model classes."""
    # Blocks
    from estimator.models.blocks.fusion_model import FusionUnet
    from estimator.models.blocks.lightweight_refiner import LightWeightRefiner
    from estimator.models.blocks.bi_directional_fusion_model import BiDirectionalFusion
    from estimator.models.blocks.position_embedding import PositionEmbeddingRandom

    # Losses
    from estimator.models.losses import SILogLoss, GradMatchLoss, SSIGMProposalLoss

    # Models
    from estimator.models.patchrefinerplus import PatchRefinerPlus
    from estimator.models.patchrefiner import PatchRefiner
    from estimator.models.patchrefiner_semi import PatchRefinerSemi

    for cls in [
        # Blocks
        FusionUnet,
        LightWeightRefiner,
        BiDirectionalFusion,
        PositionEmbeddingRandom,
        # Losses
        SILogLoss,
        GradMatchLoss,
        SSIGMProposalLoss,
        # Models
        PatchRefinerPlus,
        PatchRefiner,
        PatchRefinerSemi,
    ]:
        _MODEL_REGISTRY[cls.__name__] = cls


# Populate registry on first import
_register_all()
