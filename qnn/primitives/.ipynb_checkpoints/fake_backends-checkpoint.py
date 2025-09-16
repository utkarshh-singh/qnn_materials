# qnn/primitives/fake_backends.py
from __future__ import annotations
import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Dict, Type, Optional, Iterable

# Qiskit >= 1.0 has BackendV2 here:
try:
    from qiskit.providers.backend import BackendV2 as _BackendBase
except Exception:  # fallback for very old installs
    try:
        from qiskit.providers import BackendV2 as _BackendBase  # type: ignore
    except Exception:
        _BackendBase = object  # best-effort fallback


def _iter_modules(pkg: ModuleType) -> Iterable[ModuleType]:
    """Yield imported submodules contained in a package (safe, best-effort)."""
    if not hasattr(pkg, "__path__"):
        return
    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        name = modinfo.name
        try:
            yield importlib.import_module(name)
        except Exception:
            # skip anything that fails to import
            continue


def _collect_fake_backend_classes(pkg: ModuleType) -> Dict[str, Type]:
    """Scan a package tree for classes that look like Fake backends."""
    found: Dict[str, Type] = {}

    # include the package itself in case it defines classes at top-level
    candidates = [pkg, *list(_iter_modules(pkg))]

    for mod in candidates:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            # heuristics:
            #  - class name starts with 'Fake'
            #  - subclass of BackendV2 (if available)
            #  - lives in this package's namespace to avoid unrelated 'Fake*' classes
            if not obj.__name__.startswith("Fake"):
                continue
            if _BackendBase is not object and not issubclass(obj, _BackendBase):
                continue
            # ensure it comes from this package tree
            if not getattr(obj, "__module__", "").startswith(pkg.__name__):
                continue
            found[obj.__name__] = obj
    return found


def discover_fake_backends() -> Dict[str, Type]:
    """
    Return a dict {ClassName -> Class} of all available fake backend classes
    from any installed fake-provider namespaces.
    """
    namespaces = []
    # Newer location (qiskit-ibm-runtime)
    try:
        namespaces.append(importlib.import_module("qiskit_ibm_runtime.fake_provider"))
    except Exception:
        pass
    # Legacy location (qiskit.providers.fake_provider)
    try:
        namespaces.append(importlib.import_module("qiskit.providers.fake_provider"))
    except Exception:
        pass

    all_found: Dict[str, Type] = {}
    for ns in namespaces:
        try:
            all_found.update(_collect_fake_backend_classes(ns))
        except Exception:
            continue
    return all_found


def get_fake_backend(name: str) -> Optional[_BackendBase]:
    """
    Instantiate a fake backend by name (flexible):
      - accepts exact class name (e.g., 'FakeOslo')
      - accepts lowercase/underscored (e.g., 'fake_oslo' or 'oslo')
    Returns an instance or None if not found.
    """
    name_clean = name.strip().lower().replace("-", "_").replace(" ", "_")
    classes = discover_fake_backends()
    if not classes:
        return None

    # exact class name match first
    if name in classes:
        return classes[name]()

    # try canonical 'FakeXxx'
    canonical = "Fake" + "".join(part.capitalize() for part in name_clean.split("_") if part and part != "fake")
    if canonical in classes:
        return classes[canonical]()

    # try loose lookup (case/underscore insensitive)
    for cls_name, cls in classes.items():
        key = cls_name.lower().replace("-", "_")
        if key == name_clean or key.endswith(name_clean) or name_clean.endswith(key):
            return cls()

    return None


# # ------------- CLI test -------------
# if __name__ == "__main__":
#     classes = discover_fake_backends()
#     if not classes:
#         print("No fake backends discovered.")
#     else:
#         print(f"Discovered {len(classes)} fake backend classes:")
#         # show a few
#         for i, k in enumerate(sorted(classes)):
#             print("  -", k)
#             if i > 20:
#                 print("  ...")
#                 break

#         # quick smoke test
#         for probe in ("FakeOslo", "fake_oslo", "oslo", "FakePerth", "fake_perth", "perth"):
#             inst = get_fake_backend(probe)
#             if inst is not None:
#                 print(f"\n✔ Instantiated '{probe}':", getattr(inst, "name", type(inst).__name__))
#                 break
