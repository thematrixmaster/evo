"""COVID-19 neutralization prediction oracle.

Predicts antibody neutralization of SARS-CoV-1 and SARS-CoV-2 variants
using SRU-based RNN encoder with optional IgLM likelihood weighting.
"""

import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from .base import DifferentiableOracle, GaussianOracle
from .covid_model import AA_VOCAB, MultiABOnlyCoronavirusModel


class CovidOracle(GaussianOracle, DifferentiableOracle):
    """Oracle for predicting COVID-19 neutralization.

    Supports both SARS-CoV-1 and SARS-CoV-2 Beta variant predictions.
    Can optionally weight predictions with IgLM log-likelihood.
    Returns both the mean and variance of the distribution.

    Output:
        Neutralization logits (higher is better)

    Chain type:
        Heavy chain only (VH or VHH)

    Seed sequences:
        Dynamically loaded from CSV files in evo/oracles/data/ and scored during initialization
    """

    CHAIN_TYPE = "heavy"

    # Mapping from variant name to CSV filename
    CSV_FILES = {
        "SARSCoV1": "CovAbDab_heavy_binds SARS-CoV1.csv",
        "SARSCoV2Beta": "CovAbDab_heavy_binds SARS-CoV2_Beta.csv",
    }

    def __init__(
        self,
        variant: str = "SARSCoV1",
        weights_dir: str = "/scratch/users/stephen.lu/projects/protevo/checkpoints/oracle_weights",
        device: str = "cuda",
        use_iglm_weighting: bool = False,
        iglm_weight: Optional[float] = None,
        enable_mc_dropout: bool = False,
        mc_samples: int = 10,
        mc_dropout_seed: int = 0,
        fixed_variance: Optional[float] = None,
        precompute_seed_fitnesses: bool = True,
        cache_size: int = 10000,
    ):
        """Initialize COVID oracle.

        Parameters
        ----------
        variant : str
            Variant to predict: 'SARSCoV1' or 'SARSCoV2Beta'
        weights_dir : str
            Directory containing neut_model.ckpt
        device : str
            Device to run inference on, defaults to 'cuda'
        use_iglm_weighting : bool
            Whether to add IgLM log-likelihood weighting, defaults to True
        iglm_weight : float, optional
            Weight for IgLM term. If None, computed from std ratio
        enable_mc_dropout : bool
            Whether to use MC Dropout for uncertainty estimation, defaults to False
        mc_samples : int
            Number of forward passes for MC Dropout, defaults to 10
        mc_dropout_seed : int
            Seed for the isolated dropout RNG generator. Use the same seed across
            different oracle instances to get identical dropout masks. Defaults to 0.
            This does NOT affect external RNG state.
        fixed_variance : float, optional
            Fixed variance to use for uncertainty. If None and MC Dropout disabled,
            defaults to 1.0. Ignored if enable_mc_dropout=True.
        cache_size : int
            Maximum number of sequences to cache. Set to 0 to disable caching.
            Defaults to 10000.
        """
        super().__init__(device=device)

        if variant not in ["SARSCoV1", "SARSCoV2Beta"]:
            raise ValueError("variant must be 'SARSCoV1' or 'SARSCoV2Beta'")

        self.variant = variant
        self.variant_idx = 0 if variant == "SARSCoV1" else 1
        self.weights_dir = weights_dir
        self.use_iglm_weighting = use_iglm_weighting
        self.enable_mc_dropout = enable_mc_dropout
        self.mc_samples = mc_samples
        self.mc_dropout_seed = mc_dropout_seed
        self.precompute_seed_fitnesses = precompute_seed_fitnesses
        self.fixed_variance = fixed_variance if fixed_variance is not None else 1.0

        # Load model
        model_path = os.path.join(weights_dir, "neut_model.ckpt")
        model_args = {
            "hidden_dim": 256,
            "n_layers": 2,
            "use_srupp": False,
        }
        model_args.update(MultiABOnlyCoronavirusModel.add_extra_args())

        self.model = MultiABOnlyCoronavirusModel.load_from_checkpoint(
            model_path,
            **model_args,
        )
        self.model.to(self.device)
        if not enable_mc_dropout:
            self.model.eval()
        # If MC Dropout enabled, will toggle between train/eval dynamically

        # Initialize IgLM if requested
        self.iglm = None
        self.iglm_weight = iglm_weight
        if use_iglm_weighting:
            from iglm import IgLM

            self.iglm = IgLM()
            self.iglm_cache = {}

        # Initialize MC dropout settings
        # We'll use a separate random number generator that won't affect external RNG
        if enable_mc_dropout:
            # Create a separate random generator for dropout that won't affect global RNG
            # Set generator to same device as model
            self._dropout_generator = torch.Generator(device=self.device)
            self._dropout_generator.manual_seed(mc_dropout_seed)
            print(f"Initialized MC Dropout with {mc_samples} samples using isolated RNG (seed={mc_dropout_seed}) for deterministic predictions")

        # Initialize LRU cache for predictions
        self.cache_size = cache_size
        self._prediction_cache = OrderedDict()  # {sequence: (mean, variance)}
        if cache_size > 0:
            print(f"Initialized prediction cache with max size: {cache_size}")

        # Load seed sequences from CSV and score them
        self._seed_data = self._load_and_score_seeds()

    def _load_and_score_seeds(self):
        """Load sequences from CSV file and score them dynamically.

        Returns
        -------
        dict
            Dictionary mapping seed IDs (seed_0, seed_1, ...) to dicts containing
            'sequence' and 'fitness' keys
        """
        # Get path to CSV file
        csv_filename = self.CSV_FILES[self.variant]
        # Get the directory where this file is located
        module_dir = Path(__file__).parent
        csv_path = module_dir / "data" / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(f"Seed sequence CSV not found: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)

        # Extract sequences (first column is sequences, second is binding label)
        sequences = df.iloc[:, 0].tolist()

        print(f"Loading {len(sequences)} seed sequences from {csv_filename}...")

        # Score all sequences
        if not self.precompute_seed_fitnesses:
            # If not precomputing, set fitnesses to None
            fitnesses = [None] * len(sequences)
        else:
            fitnesses, _ = self.predict_batch(sequences)

        # Create seed_data dictionary
        seed_data = {}
        for i, (seq, fitness) in enumerate(zip(sequences, fitnesses)):
            seed_data[f"seed_{i}"] = {
                "sequence": seq,
                "fitness": float(fitness) if fitness is not None else None,
            }

        print(f"Scored {len(seed_data)} seed sequences for {self.variant}")

        return seed_data

    @property
    def seed_data(self):
        """Return the seed data dictionary with sequences and dynamically computed fitnesses."""
        return self._seed_data

    @property
    def seed_sequences(self):
        """Return list of seed sequences for backward compatibility."""
        return [data["sequence"] for data in self._seed_data.values()]

    @property
    def seed_fitnesses(self):
        """Return list of dynamically computed seed fitnesses."""
        return [data["fitness"] for data in self._seed_data.values()]

    @property
    def chain_type(self):
        """Return the antibody chain type (heavy, light, or both)."""
        return self.CHAIN_TYPE

    def get_cache_info(self):
        """Get cache statistics.

        Returns
        -------
        dict
            Dictionary with cache_size (max), cached_sequences (current count)
        """
        return {
            'max_size': self.cache_size,
            'current_size': len(self._prediction_cache),
            'usage_percent': len(self._prediction_cache) / self.cache_size * 100 if self.cache_size > 0 else 0,
        }

    def clear_cache(self):
        """Clear the prediction cache."""
        self._prediction_cache.clear()

    @property
    def higher_is_better(self):
        """Return True if higher fitness values are better, False otherwise."""
        return True  # Neutralization logits: higher is better

    def get_seed_fitness(self, seed_id: str) -> float:
        """Get dynamically computed fitness for a specific seed.

        Parameters
        ----------
        seed_id : str
            Seed identifier (e.g., 'seed_0', 'seed_1')

        Returns
        -------
        float
            Computed fitness value

        Raises
        ------
        ValueError
            If seed_id is not valid
        """
        if seed_id not in self._seed_data:
            raise ValueError(
                f"Unknown seed_id: {seed_id}. Available: {list(self._seed_data.keys())}"
            )
        fitness = self._seed_data[seed_id]["fitness"]
        if fitness is None:
            if self.precompute_seed_fitnesses:
                import warnings
                warnings.warn(
                    f"Fitness for {seed_id} is None. "
                    "This should not happen as all seed fitnesses have been precomputed."
                )
            else:
                # compute on-the-fly and save to seed_data
                sequence = self._seed_data[seed_id]["sequence"]
                fitness, _ = self.predict(sequence)
                self._seed_data[seed_id]["fitness"] = float(fitness)
        return fitness

    def _get_iglm_likelihood(self, sequence: str) -> float:
        """Get cached IgLM log-likelihood for a sequence."""
        if sequence not in self.iglm_cache:
            self.iglm_cache[sequence] = self.iglm.log_likelihood(sequence, "[HEAVY]", "[HUMAN]")
        return self.iglm_cache[sequence]

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode sequence to token IDs."""
        return torch.tensor([AA_VOCAB[aa] for aa in sequence]).long()

    def predict(self, sequence: str) -> tuple[float, float]:
        """Predict neutralization with uncertainty for a single sequence.

        Parameters
        ----------
        sequence : str
            Antibody sequence as a string

        Returns
        -------
        tuple[float, float]
            (mean, variance) of neutralization prediction
        """
        means, variances = self.predict_batch([sequence])
        return means[0], variances[0]

    def predict_batch(self, sequences: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict neutralization with uncertainty for a batch of sequences.

        Uses either MC Dropout (if enabled) or fixed variance.
        Results are cached for efficiency.

        Parameters
        ----------
        sequences : List[str]
            List of antibody sequences as strings

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (means, variances) arrays of shape (n_sequences,)
        """
        if self.cache_size == 0:
            # Caching disabled
            return self._predict_batch_uncached(sequences)

        # Check cache and separate cached vs uncached sequences
        means = []
        variances = []
        uncached_indices = []
        uncached_sequences = []

        for i, seq in enumerate(sequences):
            if seq in self._prediction_cache:
                # Cache hit - move to end (most recently used)
                self._prediction_cache.move_to_end(seq)
                mean, var = self._prediction_cache[seq]
                means.append(mean)
                variances.append(var)
            else:
                # Cache miss
                means.append(None)
                variances.append(None)
                uncached_indices.append(i)
                uncached_sequences.append(seq)

        # Predict uncached sequences
        if uncached_sequences:
            uncached_means, uncached_vars = self._predict_batch_uncached(uncached_sequences)

            # Update cache and results
            for idx, seq, mean, var in zip(uncached_indices, uncached_sequences,
                                           uncached_means, uncached_vars):
                # Add to cache
                self._prediction_cache[seq] = (float(mean), float(var))

                # Evict oldest if cache is full
                if len(self._prediction_cache) > self.cache_size:
                    self._prediction_cache.popitem(last=False)

                # Update results
                means[idx] = mean
                variances[idx] = var

        return np.array(means), np.array(variances)

    def _predict_batch_uncached(self, sequences: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict without caching (internal method)."""
        if self.enable_mc_dropout:
            return self._predict_with_mc_dropout(sequences)
        else:
            # Use standard prediction with fixed variance
            means, _ = self._predict_with_mc_dropout(sequences)
            variances = np.full_like(means, self.fixed_variance)
            return means, variances

    def _predict_with_mc_dropout(self, sequences: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict using MC Dropout for uncertainty estimation.

        Uses an isolated torch.Generator for dropout to ensure deterministic
        predictions without affecting external random state.

        Parameters
        ----------
        sequences : List[str]
            List of antibody sequences as strings

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (means, variances) arrays of shape (n_sequences,)
        """
        # Encode sequences
        encoded = [self._encode_sequence(seq) for seq in sequences]

        # Pad to same length
        max_len = max(len(seq) for seq in encoded)
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(encoded):
            padded[i, : len(seq)] = seq

        # Move to device
        padded = padded.to(self.device)

        # Keep model in eval mode - we'll manually apply dropout
        self.model.eval()

        # Run multiple forward passes with manual dropout
        all_predictions = []

        with torch.no_grad():
            for seq_idx in range(len(sequences)):
                seq_tensor = padded[seq_idx:seq_idx+1]  # (1, max_len)
                seq_predictions = []

                # Reset generator state for this sequence to ensure determinism
                self._dropout_generator.manual_seed(self.mc_dropout_seed)

                for mc_idx in range(self.mc_samples):
                    # Forward pass - we need to manually inject dropout
                    # Since the model is in eval mode, we'll use a hook to apply dropout
                    logits = self._forward_with_manual_dropout(seq_tensor)
                    pred = logits[0, self.variant_idx].float().cpu().numpy()
                    seq_predictions.append(pred)

                all_predictions.append(seq_predictions)

        # Compute mean and variance across MC samples
        # all_predictions: [[seq0_mc0, seq0_mc1, ...], [seq1_mc0, seq1_mc1, ...], ...]
        predictions_array = np.array(all_predictions).T  # (mc_samples, n_sequences)
        means = predictions_array.mean(axis=0)
        variances = predictions_array.var(axis=0)

        # Add IgLM weighting to means if requested
        # Note: IgLM is deterministic, so doesn't affect variance
        if self.use_iglm_weighting:
            iglm_scores = np.array([self._get_iglm_likelihood(seq) for seq in sequences])

            # Compute weight if not provided
            if self.iglm_weight is None:
                weight = 1.0
            else:
                weight = self.iglm_weight

            means = means + weight * iglm_scores

        return means, variances

    def _forward_with_manual_dropout(self, seq_tensor):
        """Forward pass with manual dropout using isolated generator.

        This allows us to apply dropout deterministically without affecting
        the global RNG state.
        """
        # Find the dropout module (should be the last one based on the model architecture)
        dropout_modules = [m for m in self.model.modules() if isinstance(m, torch.nn.Dropout)]

        if not dropout_modules:
            # No dropout, just do regular forward pass
            return self.model(seq_tensor)

        # Register forward hooks to apply dropout with our generator
        hooks = []

        def make_dropout_hook(dropout_prob, generator):
            def hook(module, input, output):
                if dropout_prob > 0:
                    # Apply dropout using our isolated generator
                    # Generate dropout mask
                    # change generator device to match output device
                    mask = torch.bernoulli(
                        torch.full_like(output, 1 - dropout_prob),
                        generator=generator
                    )
                    # Scale and apply mask
                    return output * mask / (1 - dropout_prob)
                return output
            return hook

        try:
            # Register hooks on dropout modules
            for dropout_mod in dropout_modules:
                if dropout_mod.p > 0:  # Only hook active dropout layers
                    hook_handle = dropout_mod.register_forward_hook(
                        make_dropout_hook(dropout_mod.p, self._dropout_generator)
                    )
                    hooks.append(hook_handle)

            # Run forward pass with hooks active
            logits = self.model(seq_tensor)

        finally:
            # Remove hooks
            for hook_handle in hooks:
                hook_handle.remove()

        return logits

    def compute_fitness_gradient(self, sequence: str) -> tuple[np.ndarray, float]:
        """Compute gradients of the predicted mean fitness w.r.t. one-hot encoding.

        This enables the Taylor-series approximation for Eq [9]:
        μ(x_j) ≈ μ(x_i) + (x_j - x_i)^T * ∇μ(x_i)

        Parameters
        ----------
        sequence : str
            Input antibody sequence

        Returns
        -------
        tuple[np.ndarray, float]
            gradient: Gradients w.r.t. one-hot encoding, shape (L, vocab_size)
            variance: Estimated variance of the prediction
        """
        # Gradient Computation (Forward Pass WITH_GRAD)
        # Encode sequence
        encoded = self._encode_sequence(sequence)
        L = len(encoded)
        vocab_size = len(AA_VOCAB)

        # Create one-hot tensor that requires gradients
        # Shape: (1, L, vocab_size)
        onehot = torch.nn.functional.one_hot(encoded, num_classes=vocab_size).float()
        onehot = onehot.unsqueeze(0).to(self.device)

        # Standard Evaluation Mode (no dropout for the mean gradient)
        self.model.eval()

        # IMPORTANT: Enable gradients even if called from torch.no_grad() context
        with torch.enable_grad():
            onehot.requires_grad_(True)

            # --- The Embedding Trick ---
            # Bypass the standard embedding layer (which takes ints) and manually
            # multiply one-hots with the embedding weight matrix.
            embed_weight = self.model.encoder.seq_embedding.weight
            embeds = torch.matmul(onehot, embed_weight)  # (1, L, embed_dim)

            # Pass through the rest of the model manually
            # Note: We must replicate the forward pass logic of MultiABOnlyCoronavirusModel
            padding_mask = torch.ones(1, L, dtype=torch.bool, device=self.device)

            # Encoder (RNN/SRU)
            hidden, _ = self.model.encoder.rnn_ab(embeds, padding_mask=padding_mask)

            # Mean Pooling
            pooled = hidden.mean(dim=1)

            # Neutralization Head
            logits = self.model.fc_neut(pooled)

            # Select the specific variant we are guiding towards
            predicted_mean = logits[0, self.variant_idx]

            # Backward pass to get gradients of the MEAN prediction
            predicted_mean.backward()

            if onehot.grad is None:
                raise RuntimeError("Gradients failed to compute. Check model graph.")

            # Extract gradients: (L, vocab_size)
            grad_matrix = onehot.grad[0].cpu().numpy()

        # Get variance estimate
        if self.enable_mc_dropout:
            _, variances = self._predict_with_mc_dropout([sequence])
            variance = variances[0]
        else:
            variance = self.fixed_variance

        return grad_matrix, variance

    def __repr__(self):
        iglm_str = f", use_iglm_weighting={self.use_iglm_weighting}"
        return f"CovidOracle(variant='{self.variant}', device='{self.device}'{iglm_str})"


if __name__ == "__main__":
    oracle = CovidOracle(variant="SARSCoV1", device="cuda")
    print(oracle.predict("QVQLQESGPGLVKPSETLSLTCTVSGGFIGPHYWSWVRQPPGKGLEWIGYIYISGSTNYNPSLKSRLTISVDMSKSQFSLTLSSATAADTAVYYCARGGGYLETGPFEYWGQGTLVTVSS"))
    breakpoint()
    