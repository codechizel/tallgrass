"""Exploratory Graph Analysis (EGA) — network psychometrics for dimensionality assessment.

Implements Hudson Golino's EGA framework in Python:
- Tetrachoric correlations for binary vote data
- GLASSO + EBIC network estimation
- Community detection (Walktrap / Leiden)
- Bootstrap stability assessment (bootEGA)
- Total Entropy Fit Index (TEFI)
- Unique Variable Analysis (UVA)

See docs/network-psychometrics-ega-deep-dive.md for background.
"""

from analysis.ega.boot_ega import BootEGAResult, run_boot_ega
from analysis.ega.community import CommunityResult, detect_communities
from analysis.ega.ega import EGAResult, run_ega
from analysis.ega.glasso import GLASSOResult, glasso_ebic
from analysis.ega.tefi import compute_tefi
from analysis.ega.tetrachoric import tetrachoric_corr_matrix
from analysis.ega.uva import UVAResult, run_uva

__all__ = [
    "BootEGAResult",
    "CommunityResult",
    "EGAResult",
    "GLASSOResult",
    "UVAResult",
    "compute_tefi",
    "detect_communities",
    "glasso_ebic",
    "run_boot_ega",
    "run_ega",
    "run_uva",
    "tetrachoric_corr_matrix",
]
