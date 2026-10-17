"""Training exports load lazily so offline converters do not import the training stack."""

import importlib

__all__ = ["LOADERS", "DevicePrefetcher", "build_split", "collate", "dataset_of", "horizon", "import_plugins",
           "loader_name", "make_loader", "register"]
_loaded = False


def _load_defaults():
    global _loaded
    if _loaded:
        return
    registry = importlib.import_module("unidrivevla.data.registry")
    for name in ("bench2drive", "mixture", "multidomain", "navsim", "nuscenes"):
        importlib.import_module(f"unidrivevla.data.{name}")
    # These modules also run in SDK-only conversion environments; registration belongs to training.
    for name in ("physicalai", "lead"):
        module = importlib.import_module(f"unidrivevla.data.{name}")
        registry.register(name)(getattr(module, f"{name}_loader"))
    for name in __all__:
        if name != "DevicePrefetcher":
            globals()[name] = getattr(registry, name)
    globals()["DevicePrefetcher"] = importlib.import_module("unidrivevla.data.prefetch").DevicePrefetcher
    _loaded = True


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    _load_defaults()
    return globals()[name]
