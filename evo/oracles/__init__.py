"""Protein fitness oracles for evolutionary inference.

This module provides a unified interface for protein fitness prediction
using various oracle models extracted from the CloneBO framework.

Available Oracles:
    - LLMOracle: Language model-based oracle using clonal family data
    - CovidOracle: SARS-CoV-1/CoV-2 neutralization prediction
    - RandOracle: Hybrid oracle with LLM and random ByteNet

Oracle Properties:
    All oracles provide the following properties:
    - seed_sequences: Default starting sequences used in CloneBO optimization
    - chain_type: Antibody chain type ("heavy", "light", or "both")
    - higher_is_better: Boolean indicating fitness direction

Example usage:
    >>> from evo.oracles import get_oracle
    >>>
    >>> # Load LLM oracle
    >>> oracle = get_oracle("clone")
    >>> fitness = oracle("EVQLVESGGGLVQPGGSLR...")
    >>>
    >>> # Check oracle properties
    >>> print(oracle.chain_type)  # "heavy"
    >>> print(oracle.higher_is_better)  # False (lower loss is better)
    >>> print(oracle.seed_sequences)  # List of seed sequences
    >>>
    >>> # Load COVID oracle
    >>> oracle = get_oracle("SARSCoV1")
    >>> fitness = oracle("EVQLVESGGGLVQPGGSLR...")
    >>> print(oracle.higher_is_better)  # True (higher neutralization is better)
    >>>
    >>> # Batch prediction
    >>> sequences = ["EVQL...", "QVQL...", "DVQL..."]
    >>> fitnesses = oracle(sequences)
"""

from typing import Union

from .base import BaseOracle, DifferentiableOracle, GaussianOracle
from .covid_oracle import CovidOracle
from .llm_oracle import LLMOracle
from .rand_oracle import RandOracle

__all__ = [
    "BaseOracle",
    "DifferentiableOracle",
    "GaussianOracle",
    "LLMOracle",
    "CovidOracle",
    "RandOracle",
    "get_oracle",
]


def get_oracle(
    oracle_name: str,
    device: str = "cuda",
    weights_dir: str = "/scratch/users/stephen.lu/projects/protevo/checkpoints/oracle_weights",
    **kwargs,
) -> BaseOracle:
    """Factory function to instantiate protein fitness oracles.

    Parameters
    ----------
    oracle_name : str
        Name of the oracle to load:
        - "clone": LLM oracle trained on clonal families
        - "SARSCoV1": SARS-CoV-1 neutralization oracle
        - "SARSCoV2Beta": SARS-CoV-2 Beta neutralization oracle
        - "rand_X": Random oracle with mixing parameter X (e.g., "rand_0.5")
    device : str
        Device to run inference on ('cuda' or 'cpu'), defaults to 'cuda'
    weights_dir : str
        Directory containing oracle model weights
    **kwargs
        Additional keyword arguments passed to oracle constructor

    Returns
    -------
    BaseOracle
        Initialized oracle instance

    Raises
    ------
    ValueError
        If oracle_name is not recognized

    Examples
    --------
    >>> # Load LLM oracle
    >>> oracle = get_oracle("clone")
    >>> fitness = oracle("EVQLVESGGGLVQPGGSLR...")

    >>> # Load COVID oracle with custom settings
    >>> oracle = get_oracle("SARSCoV1", use_iglm_weighting=False)
    >>> fitness = oracle("EVQLVESGGGLVQPGGSLR...")

    >>> # Load random oracle with alpha=0.3
    >>> oracle = get_oracle("rand_0.3")
    >>> fitness = oracle("EVQLVESGGGLVQPGGSLR...")
    """
    if oracle_name == "clone":
        return LLMOracle(weights_dir=weights_dir, device=device, **kwargs)

    elif oracle_name in ["SARSCoV1", "SARSCoV2Beta"]:
        return CovidOracle(variant=oracle_name, weights_dir=weights_dir, device=device, **kwargs)

    elif oracle_name.startswith("rand_"):
        try:
            alpha = float(oracle_name.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid rand oracle name: {oracle_name}. "
                "Expected format: 'rand_X' where X is a float (e.g., 'rand_0.5')"
            )
        return RandOracle(alpha=alpha, weights_dir=weights_dir, device=device, **kwargs)

    else:
        raise ValueError(
            f"Unknown oracle: {oracle_name}. "
            "Available oracles: 'clone', 'SARSCoV1', 'SARSCoV2Beta', 'rand_X'"
        )
