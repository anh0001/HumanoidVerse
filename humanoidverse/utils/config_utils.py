import math
from loguru import logger
from omegaconf import OmegaConf


def _register_resolver(name, fn):
    """Register Hydra resolver once; skip duplicates to avoid noisy warnings."""
    has_resolver = getattr(OmegaConf, "has_resolver", None)
    try:
        if callable(has_resolver) and has_resolver(name):
            logger.debug(f"Resolver '{name}' already registered; skipping duplicate.")
            return
        OmegaConf.register_new_resolver(name, fn)
    except Exception as exc:  # pragma: no cover - defensive logging
        if "already registered" in str(exc):
            logger.debug(f"Resolver '{name}' already registered; skipping duplicate.")
            return
        logger.warning(f"Failed to register resolver '{name}': {exc}")


_RESOLVERS = {
    "eval": eval,
    "if": lambda pred, a, b: a if pred else b,
    "eq": lambda x, y: x.lower() == y.lower(),
    "sqrt": lambda x: math.sqrt(float(x)),
    "sum": lambda x: sum(x),
    "ceil": lambda x: math.ceil(x),
    "int": lambda x: int(x),
    "len": lambda x: len(x),
    "sum_list": lambda lst: sum(lst),
}

for _name, _fn in _RESOLVERS.items():
    _register_resolver(_name, _fn)
