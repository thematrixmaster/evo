"""TAG weight predictor classes for CTMC guided sampling.

Each predictor maps delta_mu (L, V) — the predicted fitness change for every possible
mutation — to guidance weights (L, V) that multiply the CTMC rate matrix entries.

Usage::

    from evo.oracles.tag_predictors import EnergyTAGPredictor
    predictor = EnergyTAGPredictor(guidance_strength=50.0)
    # pass as tag_weight_predictor= to generate_with_gillespie
"""

import abc

import torch
from torch import Tensor


class TAGWeightPredictor(abc.ABC):
    """Abstract base: maps delta_mu (L, V) → guidance weights (L, V)."""

    @abc.abstractmethod
    def compute_weights(self, delta_mu: Tensor) -> Tensor:
        """Return per-token guidance weights from predicted fitness changes.

        Parameters
        ----------
        delta_mu : Tensor, shape (L, V)
            Predicted fitness change for mutating each position to each token.
            delta_mu[l, v] = grad[l, v] - grad[l, v_curr].

        Returns
        -------
        weights : Tensor, shape (L, V)
            Multiplicative weights for Q[l, v_curr, v]. Values > 1 boost the
            rate, values < 1 suppress it. Must be non-negative.
        """
        ...


class GaussianTAGPredictor(TAGWeightPredictor):
    """Gaussian noise model: weight = (2·Φ(Δμ/σ))^γ.

    Normalises delta_mu by an adaptive σ (mean |Δμ|) before applying the
    standard-normal CDF.  Weights are bounded in [0, 2^γ].

    Parameters
    ----------
    guidance_strength : float
        Exponent γ.  Higher values sharpen the preference for predicted
        beneficial mutations.
    """

    def __init__(self, guidance_strength: float = 5.0, sigma: float | Tensor | None = None):
        self.guidance_strength = guidance_strength
        self.sigma = sigma

    def compute_weights(self, delta_mu: Tensor) -> Tensor:
        if self.sigma is not None:
            sigma = torch.as_tensor(self.sigma, device=delta_mu.device, dtype=delta_mu.dtype)
        else:
            nonzero_mask = delta_mu.abs() > 1e-12
            if nonzero_mask.any():
                sigma = delta_mu[nonzero_mask].abs().mean()
            else:
                sigma = torch.tensor(1e-8, device=delta_mu.device, dtype=delta_mu.dtype)
        sigma = sigma.clamp(min=1e-8)
        probs = torch.special.ndtr(delta_mu / sigma)
        return (2.0 * probs).pow(self.guidance_strength)


class EnergyTAGPredictor(TAGWeightPredictor):
    """Energy-based model p(y) ∝ exp(r(y)): weight = exp(γ·Δμ).

    Derived from the Doob h-transform for the energy-based distribution:

        Q_guided[v_curr → v_new] = Q[v_curr → v_new] · exp(r(y_new) − r(y_curr))
                                 ≈ Q[v_curr → v_new] · exp(Δμ)   (TAG approx.)

    `guidance_strength` acts as inverse temperature γ.  The exponent is clamped
    to [−20, 20] for numerical stability (exp(20) ≈ 5e8).

    Unlike GaussianTAGPredictor, this form uses gradient magnitudes directly
    without σ-normalisation, so `guidance_strength` must be tuned to match the
    scale of delta_mu values returned by the oracle.

    Parameters
    ----------
    guidance_strength : float
        Inverse temperature γ.  A reasonable starting point is
        ``guidance_strength ≈ 10 / mean(|delta_mu|)`` so the top mutations
        receive roughly exp(10) ≈ 22k× boost.
    """

    def __init__(self, guidance_strength: float = 1.0):
        self.guidance_strength = guidance_strength

    def compute_weights(self, delta_mu: Tensor) -> Tensor:
        return torch.exp((self.guidance_strength * delta_mu).clamp(-20.0, 20.0))
