"""ESMFold2-based oracle for antibody-antigen binding optimization.

Folds the heavy + light + antigen complex with ESMFold2 and uses the
Ab-Ag chain-pair ipTM as the primary binding signal.  Optionally also
uses the normalized interface PAE (Germinal convention).

Architecture
------------
The oracle spawns ``esmfold2_fold_server.py`` (via the ESMFold2 venv) as a
persistent subprocess on the first call.  The model is loaded once;
subsequent requests are cheap I/O.  Results are LRU-cached.

Scoring modes
-------------
ab_ag_iptm       mean(H-A ipTM, L-A ipTM)          higher = better
interface_pae_norm  1/(1+(PAE/10)²) at Ab↔Ag       higher = better
combined         0.5 * ab_ag_iptm + 0.5 * interface_pae_norm

Taylor guidance (DifferentiableOracle)
---------------------------------------
``compute_fitness_gradient(heavy_seq)`` runs a single differentiable
trunk pass through ESMFold2ExperimentalModel with ``res_type_soft``.
The gradient of the mean Ab-Ag inter-contact probability w.r.t. the
heavy-chain residue-type one-hot is returned as a ``(L, 25)`` array
in the shared oracle vocabulary (same ordering as covid_model.AA_VOCAB).
"""

import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import DifferentiableOracle, GaussianOracle

_FOLD_SERVER = str(Path(__file__).resolve().with_name("esmfold2_fold_server.py"))
_ESMFOLD2_PYTHON = "/scratch/users/stephen.lu/envs/esmfold2/bin/python"

_VALID_SCORING = ("ab_ag_iptm", "interface_pae_norm", "combined")

# ESMFold2 res_type vocabulary (see esm/models/esmfold2/constants.py)
# Standard amino acids occupy indices 2-21 (ALA=2 … VAL=21)
# The shared oracle vocab (covid_model.AA_VOCAB) has these at indices 1-20
# with the mapping aa_vocab_idx = res_type_idx - 1.
_ESM_AA_START = 2   # index of ALA in ESMFold2 res_type vocab
_ESM_AA_END   = 22  # exclusive end (VAL = 21)
_ORACLE_VOCAB_SIZE = 25  # len(covid_model.AA_VOCAB)
_ORACLE_AA_START = 1     # index of A in oracle vocab


class ESMFold2Oracle(GaussianOracle, DifferentiableOracle):
    """Oracle that scores Ab-Ag binding using ESMFold2 structural confidence.

    Optimises the heavy chain while holding the light chain and antigen
    fixed.  Higher score = better predicted binding.

    Implements both ``GaussianOracle`` (for exact guidance via single-mutant
    enumeration) and ``DifferentiableOracle`` (for Taylor-approximated guidance
    via one differentiable trunk pass per step).

    Parameters
    ----------
    light_chain : str
        Fixed light-chain amino acid sequence.
    antigen : str
        Fixed antigen amino acid sequence.
    seed_sequences : list[str]
        Initial heavy-chain sequences used to seed optimisation.
    scoring_fn : str
        Which score to return as the fitness signal.
        One of ``"ab_ag_iptm"``, ``"interface_pae_norm"``, ``"combined"``.
    fixed_variance : float
        Variance returned alongside every prediction (ESMFold2 is
        deterministic, so variance is not estimated empirically).
    num_loops : int
        ESMFold2 recycling iterations (default 10).
    num_steps : int
        Diffusion sampling steps for scoring (default 50).
    seed : int
        RNG seed for reproducible diffusion sampling.
    cache_size : int
        Max number of (sequence → score) entries to keep in the LRU cache.
    venv_python : str
        Path to the ESMFold2 venv Python interpreter.
    """

    CHAIN_TYPE = "heavy"

    def __init__(
        self,
        light_chain: str,
        antigen: str,
        seed_sequences: Optional[List[str]] = None,
        device: str = "cuda",
        scoring_fn: str = "ab_ag_iptm",
        fixed_variance: float = 0.01,
        num_loops: int = 10,
        num_steps: int = 50,
        seed: int = 0,
        cache_size: int = 1000,
        venv_python: str = _ESMFOLD2_PYTHON,
    ):
        # GaussianOracle.__init__ sets self.device
        GaussianOracle.__init__(self, device=device)

        if scoring_fn not in _VALID_SCORING:
            raise ValueError(f"scoring_fn must be one of {_VALID_SCORING}, got {scoring_fn!r}")

        self.light_chain = light_chain
        self.antigen = antigen
        self._seed_sequences = seed_sequences or []
        self.scoring_fn = scoring_fn
        self.fixed_variance = fixed_variance
        self.num_loops = num_loops
        self.num_steps = num_steps
        self.fold_seed = seed
        self.cache_size = cache_size
        self.venv_python = venv_python

        self._prediction_cache: OrderedDict = OrderedDict()
        self._server_proc: Optional[subprocess.Popen] = None
        self._seed_fitnesses_cache: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Subprocess server management
    # ------------------------------------------------------------------

    def _start_server(self):
        print("[ESMFold2Oracle] Starting fold server…", file=sys.stderr, flush=True)
        self._server_proc = subprocess.Popen(
            [self.venv_python, _FOLD_SERVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        # Drain stdout until we see the "READY" sentinel.
        # The Biohub transformers fork emits 🚨 warnings to stdout before the
        # model finishes loading, so we skip non-READY lines.
        for line in self._server_proc.stdout:
            stripped = line.strip()
            if stripped == "READY":
                break
            print(f"[ESMFold2Oracle][server] {stripped}", file=sys.stderr)
        else:
            self._server_proc.kill()
            raise RuntimeError("[ESMFold2Oracle] Server exited before sending READY.")
        print("[ESMFold2Oracle] Fold server ready.", file=sys.stderr, flush=True)

    def _ensure_server(self):
        if self._server_proc is None or self._server_proc.poll() is not None:
            self._start_server()

    # ------------------------------------------------------------------
    # Low-level IPC helpers
    # ------------------------------------------------------------------

    def _send_request(self, req: dict) -> dict:
        """Send one JSON request to the server; return the parsed JSON response."""
        self._server_proc.stdin.write(json.dumps(req) + "\n")
        self._server_proc.stdin.flush()
        for line in self._server_proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                print(f"[ESMFold2Oracle][server] {line}", file=sys.stderr)
        raise RuntimeError("[ESMFold2Oracle] Server closed stdout before returning a response.")

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    def _score_from_response(self, resp: dict) -> float:
        if "error" in resp:
            raise RuntimeError(f"[ESMFold2Oracle] Server error: {resp['error']}")
        if self.scoring_fn == "ab_ag_iptm":
            return resp["ab_ag_iptm"]
        elif self.scoring_fn == "interface_pae_norm":
            return resp["interface_pae_norm"]
        else:  # combined
            return 0.5 * resp["ab_ag_iptm"] + 0.5 * resp["interface_pae_norm"]

    def _fold_one(self, heavy: str) -> dict:
        """Send one score request to the server and return raw metric dict."""
        return self._send_request({
            "heavy": heavy,
            "light": self.light_chain,
            "antigen": self.antigen,
            "num_loops": self.num_loops,
            "num_steps": self.num_steps,
            "seed": self.fold_seed,
        })

    # ------------------------------------------------------------------
    # GaussianOracle interface
    # ------------------------------------------------------------------

    def predict(self, sequence: str) -> Tuple[float, float]:
        means, variances = self.predict_batch([sequence])
        return float(means[0]), float(variances[0])

    def predict_batch(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_server()

        means: list = [None] * len(sequences)
        variances: list = [None] * len(sequences)
        uncached: list = []

        for i, seq in enumerate(sequences):
            if seq in self._prediction_cache:
                self._prediction_cache.move_to_end(seq)
                means[i], variances[i] = self._prediction_cache[seq]
            else:
                uncached.append(i)

        for i in uncached:
            resp = self._fold_one(sequences[i])
            score = self._score_from_response(resp)
            means[i] = score
            variances[i] = self.fixed_variance

            self._prediction_cache[sequences[i]] = (score, self.fixed_variance)
            if len(self._prediction_cache) > self.cache_size:
                self._prediction_cache.popitem(last=False)

        return np.array(means, dtype=float), np.array(variances, dtype=float)

    # ------------------------------------------------------------------
    # DifferentiableOracle interface — Taylor guidance
    # ------------------------------------------------------------------

    def compute_fitness_gradient(self, sequence: str) -> Tuple[np.ndarray, float]:
        """Gradient of Biohub structural fitness w.r.t. heavy-chain one-hot.

        structural_fitness = −(0.5·intra_contact + 0.5·inter_contact + 0.2·glob)
          intra_contact : binned entropy within 14 Å, H+L internal, k=2, min_sep=9
          inter_contact : binned entropy within 22 Å, H+L→Ag, k=1
          glob          : ELU(Rg − 2.38 × n^0.365) for H+L sub-complex

        Uses a single differentiable trunk pass through ESMFold2-Experimental-Fast
        (``res_type_soft`` pathway).  The returned gradient is in the shared oracle
        vocabulary (25 tokens, same ordering as ``covid_model.AA_VOCAB``):

          index 0  = '#'  (unused, 0-filled)
          index 1  = 'A'  … index 20 = 'V'  (20 standard AAs)
          index 21–24 = 'X', '-', 'O', '*'  (unused, 0-filled)

        This format lets the CTMC Taylor guidance code in ``ctmc.py`` map gradient
        entries to CTMC vocabulary tokens without changes.

        Parameters
        ----------
        sequence : str
            Heavy-chain amino acid sequence to differentiate at.

        Returns
        -------
        grad_matrix : np.ndarray, shape (L, 25)
            d(inter_contact) / d(one_hot[l, v]) for each position l and oracle
            vocab token v.
        variance : float
            Fixed variance estimate (same as returned by predict()).
        """
        self._ensure_server()

        # Support paired sequences encoded as "VH.VL" (CTMC chain-break convention).
        # If "." is present, split into heavy and light; otherwise use self.light_chain.
        is_paired = "." in sequence
        if is_paired:
            parts = sequence.split(".", 1)
            heavy, light = parts[0], parts[1]
        else:
            heavy, light = sequence, self.light_chain

        resp = self._send_request({
            "type": "gradient",
            "heavy": heavy,
            "light": light,
            "antigen": self.antigen,
            "num_loops": self.num_loops,
            "seed": self.fold_seed,
        })

        if "error" in resp:
            raise RuntimeError(f"[ESMFold2Oracle] Gradient server error: {resp['error']}")

        # Cache the structural fitness returned alongside the gradient so callers
        # can track the actual optimization objective without an extra server call.
        self.last_structural_fitness = float(resp.get("structural_fitness", float("nan")))

        # Server returns grad_h (L_heavy, 33) and grad_l (L_light, 33) in ESMFold2
        # res_type vocab.  Map res_type[2..21] → oracle vocab[1..20].
        grad_h_esm = np.array(resp["grad_h"], dtype=np.float32)  # (L_H, 33)
        L_heavy = grad_h_esm.shape[0]

        if is_paired:
            grad_l_esm = np.array(resp["grad_l"], dtype=np.float32)  # (L_L, 33)
            L_light = grad_l_esm.shape[0]
            grad_oracle = np.zeros((L_heavy + L_light, _ORACLE_VOCAB_SIZE), dtype=np.float32)
            grad_oracle[:L_heavy, _ORACLE_AA_START:_ORACLE_AA_START + 20] = (
                grad_h_esm[:, _ESM_AA_START:_ESM_AA_END]
            )
            grad_oracle[L_heavy:, _ORACLE_AA_START:_ORACLE_AA_START + 20] = (
                grad_l_esm[:, _ESM_AA_START:_ESM_AA_END]
            )
        else:
            grad_oracle = np.zeros((L_heavy, _ORACLE_VOCAB_SIZE), dtype=np.float32)
            grad_oracle[:, _ORACLE_AA_START:_ORACLE_AA_START + 20] = (
                grad_h_esm[:, _ESM_AA_START:_ESM_AA_END]
            )

        return grad_oracle, self.fixed_variance

    # ------------------------------------------------------------------
    # Oracle properties
    # ------------------------------------------------------------------

    @property
    def chain_type(self) -> str:
        return self.CHAIN_TYPE

    @property
    def higher_is_better(self) -> bool:
        return True

    @property
    def seed_sequences(self) -> List[str]:
        return self._seed_sequences

    @property
    def seed_fitnesses(self) -> List[float]:
        if self._seed_fitnesses_cache is None:
            if not self._seed_sequences:
                self._seed_fitnesses_cache = []
            else:
                fitnesses, _ = self.predict_batch(self._seed_sequences)
                self._seed_fitnesses_cache = fitnesses.tolist()
        return self._seed_fitnesses_cache

    def predict_with_light(self, sequence: str, light: str) -> Tuple[float, float]:
        """Score with an explicit light chain (for codesign evaluation).

        Unlike predict(), this bypasses the cache and uses the provided light
        chain instead of self.light_chain.
        """
        self._ensure_server()
        resp = self._send_request({
            "heavy": sequence,
            "light": light,
            "antigen": self.antigen,
            "num_loops": self.num_loops,
            "num_steps": self.num_steps,
            "seed": self.fold_seed,
        })
        return self._score_from_response(resp), self.fixed_variance

    def predict_scfv(self, scfv: str, num_loops: int = 3, num_steps: int = 200) -> Tuple[float, float]:
        """Score a scFv (VH-linker-VL) against the antigen; returns global iptm.

        Uses a 2-chain (Ag | scFv) fold matching binder_design.py's hero critic
        evaluation for VH/VL codesign. Default num_loops=3, num_steps=200.
        """
        self._ensure_server()
        resp = self._send_request({
            "type": "score_scfv",
            "scfv": scfv,
            "antigen": self.antigen,
            "num_loops": num_loops,
            "num_steps": num_steps,
            "seed": self.fold_seed,
        })
        if "error" in resp:
            raise RuntimeError(f"[ESMFold2Oracle] ScFv score error: {resp['error']}")
        return resp["iptm"], self.fixed_variance

    def get_cache_info(self) -> dict:
        return {
            "max_size": self.cache_size,
            "current_size": len(self._prediction_cache),
            "usage_percent": (
                len(self._prediction_cache) / self.cache_size * 100
                if self.cache_size > 0 else 0
            ),
        }

    def clear_cache(self):
        self._prediction_cache.clear()

    def __del__(self):
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.stdin.close()
            self._server_proc.wait()

    def __repr__(self):
        return (
            f"ESMFold2Oracle(scoring_fn={self.scoring_fn!r}, "
            f"num_loops={self.num_loops}, num_steps={self.num_steps})"
        )
