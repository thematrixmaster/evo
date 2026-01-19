"""Base oracle class for protein fitness prediction.

Defines the common interface for all protein fitness oracles.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np


class BaseOracle(ABC):
    """Abstract base class for protein fitness oracles.

    All oracle implementations should inherit from this class and implement
    the predict() and predict_batch() methods.

    The oracle should be callable, allowing predictions via:
        fitness = oracle(sequence)
        fitnesses = oracle(sequences)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the oracle.

        Parameters
        ----------
        device : str
            Device to run inference on ('cuda' or 'cpu'), defaults to 'cuda'
        """
        self.device = device

    @abstractmethod
    def predict(self, sequence: str) -> float:
        """Predict fitness for a single protein sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence as a string

        Returns
        -------
        float
            Predicted fitness score
        """

    @abstractmethod
    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """Predict fitness for a batch of protein sequences.

        Parameters
        ----------
        sequences : List[str]
            List of protein sequences as strings

        Returns
        -------
        np.ndarray
            Array of predicted fitness scores, shape (n_sequences,)
        """

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"


class GaussianOracle(BaseOracle):
    """Abstract base class for Gaussian oracles.
    
    All Gaussian oracles should inherit from this class and implement
    the predict() and predict_batch() methods which now return both the mean and variance of the distribution.
    This is in contrast to the BaseOracle class which only returns the mean.
    """

    @abstractmethod
    def predict(self, sequence: str) -> Tuple[float, float]:
        """Predict fitness for a single protein sequence with uncertainty.
        Parameters
        ----------
        sequence : str
            Protein sequence as a string
        Returns
        -------
        Tuple[float, float]
            Tuple of mean fitness score and variance score
        """
        pass

    @abstractmethod
    def predict_batch(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fitness for a batch of protein sequences with uncertainty.
        Parameters
        ----------
        sequences : List[str]
            List of protein sequences as strings
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of mean fitness scores and variance scores, shape (n_sequences,), (n_sequences,)
        """
        pass


class DifferentiableOracle(BaseOracle):
    """Abstract base class for differentiable oracles.

    All differentiable oracles should inherit from this class and implement
    the compute_fitness_gradient() method.
    """

    @abstractmethod
    def compute_fitness_gradient(self, sequence: str) -> np.ndarray:
        """Compute gradients of the predicted fitness w.r.t. one-hot encoding.
        """
        