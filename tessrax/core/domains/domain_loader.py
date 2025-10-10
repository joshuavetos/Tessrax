"""
Tessrax Domain Loader
---------------------

Dynamically discovers and loads all domain-specific detectors implementing
DomainInterface from the /domains/ directory.
"""

import pkgutil
import importlib
import inspect
from typing import Dict
from tessrax.core.interfaces import DomainInterface


def load_domains() -> Dict[str, DomainInterface]:
    """Discover and instantiate available domain modules."""
    domains = {}
    for _, modname, _ in pkgutil.iter_modules(["domains"]):
        try:
            module = importlib.import_module(f"domains.{modname}.{modname}_contradiction_detector")
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, DomainInterface) and cls is not DomainInterface:
                    instance = cls()
                    domains[instance.name] = instance
        except Exception as e:
            print(f"⚠️ Failed to load domain {modname}: {e}")
    return domains