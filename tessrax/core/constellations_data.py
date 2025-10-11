"""
Constellation Blueprint Registry
--------------------------------
Canonical definitions for mythic constellations.
Each entry: name, capability unlocked, required mythics.
"""

from .constellations import Constellation

CONSTELLATION_DB = [
    Constellation(
        name="The Airplane",
        capability="Flight",
        mythics_required=[
            "Heavier Than Air",
            "Control vs Stability",
            "Power vs Weight",
            "Theory vs Practice",
            "Progress vs Survival"
        ]
    ),
    # Placeholder for future constellations:
    # Constellation("The Builder","Creation",[...]),
    # Constellation("The Communicator","Translation",[...]),
]