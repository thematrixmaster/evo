"""
Persistent ESMFold2 fold server.

Loads ESMFold2ExperimentalModel once at startup then serves fold/gradient requests
from stdin indefinitely.

Protocol (newline-delimited JSON):

  3-chain score request  (CTMC gradient guidance):
    {"heavy": str, "light": str, "antigen": str,
     "num_loops": int, "num_steps": int, "seed": int}
  3-chain score response:
    {"ab_ag_iptm": float, "h_ag_iptm": float, "l_ag_iptm": float,
     "iptm": float, "ptm": float,
     "interface_pae": float, "interface_pae_norm": float,
     "mean_plddt": float}

  2-chain scFv score request  (final VH/VL codesign evaluation):
    {"type": "score_scfv", "scfv": str, "antigen": str,
     "num_loops": int, "num_steps": int, "seed": int}
  2-chain scFv score response:
    {"iptm": float, "ptm": float, "mean_plddt": float}

  Gradient request:
    {"type": "gradient", "heavy": str, "light": str, "antigen": str,
     "num_loops": int, "seed": int}
  Gradient response:
    {"grad_h": [[float, ...], ...],  # shape (L_heavy, 33)
     "grad_l": [[float, ...], ...],  # shape (L_light, 33)
     "structural_fitness": float}    # = -(0.5*intra + 0.5*inter + 0.2*glob)

  Error:    {"error": str}

Uses ESMFold2ExperimentalModel (supports res_type_soft → differentiable distogram).
Model: biohub/ESMFold2-Experimental-Fast (24 trunk layers, no MSA, 128-bin distogram).
This is the model Biohub designed for gradient-based sequence optimization.

Gradient objective: Biohub structural fitness = −(0.5·intra_contact + 0.5·inter_contact + 0.2·glob)
  - intra_contact : binned entropy within 14 Å, H+L internal, k=2 contacts, min_sep=9
  - inter_contact : binned entropy within 22 Å, H+L→Ag, k=1 contact
  - glob          : ELU(Rg − 2.38 × n^0.365) for H+L sub-complex
Negated so gradient points toward fitness improvement (CTMC sampler maximizes).

Signals readiness by printing "READY\\n" to stdout after model load.

Must be run with the ESMFold2 venv:
  /scratch/users/stephen.lu/envs/esmfold2/bin/python evo/evo/oracles/esmfold2_fold_server.py
"""

import os

# All models live in YSCRATCH HF hub cache.
# HF_HOME may already be set by the shell; if not, default to YSCRATCH.
# TRANSFORMERS_CACHE is always forced to $HF_HOME/hub — the shell sometimes points it
# to the old $HF_HOME/transformers/ path, which is the wrong cache layout for huggingface_hub.
_YSCRATCH_HF = "/scratch/users/spa-evolution-yss/huggingface-cache"
os.environ.setdefault("HF_HOME", _YSCRATCH_HF)
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Distogram configuration (biohub/ESMFold2-Experimental-Fast)
_DISTO_MIN = 2.0    # Å
_DISTO_MAX = 52.0   # Å
_DISTO_BINS = 128


# ---- Biohub structural loss helpers (Algorithms 12, 13) ----------------------

def _get_mid_points(device: str) -> torch.Tensor:
    """128 bin midpoints for the 2–52 Å distogram."""
    boundaries = torch.linspace(2.0, 52.0, 127)
    exp_bounds = torch.cat([torch.tensor([1.0]), boundaries, torch.tensor([57.0])])
    return ((exp_bounds[:-1] + exp_bounds[1:]) / 2).to(device)


def _binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Entropy of the distance distribution restricted to bins below cutoff.

    Asymmetric design (from Biohub binder_design.py):
      px uses softmax over masked logits (renormalized within cutoff);
      log_px uses log_softmax over original logits.
    """
    bin_mask = ~(bin_distance < cutoff)
    px = torch.softmax(dgram - 1e7 * bin_mask, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def _masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the k smallest values under mask along the last dim."""
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1], device=y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0.0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def _masked_avg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along last axis."""
    mask = mask.bool()
    return torch.where(mask, x, 0.0).sum(-1) / (mask.sum(-1).float() + 1e-8)


def _contact_loss(dgram: torch.Tensor, bin_distance: torch.Tensor,
                   k: int, min_sep: int, cutoff: float,
                   chain_mask: torch.Tensor, binder_mask: torch.Tensor) -> torch.Tensor:
    """Algorithm 12: entropy-based contact loss.

    dgram        : (L, L, 128)
    chain_mask   : (L,) float — residues to average over (rows)
    binder_mask  : (L,) float — eligible column residues (source of contacts)
    """
    con = _binned_entropy(dgram, bin_distance, cutoff)  # (L, L)
    if min_sep > 0:
        pos = torch.arange(dgram.shape[0], device=dgram.device)
        sep = (pos[:, None] - pos[None, :]).abs() >= min_sep   # (L, L)
        binder_mask = torch.logical_and(sep, binder_mask.bool())
    per_res = _masked_min_k(con, mask=binder_mask, k=k)   # (L,)
    return _masked_avg(per_res, mask=chain_mask)           # scalar


def _glob_loss(dgram_ab: torch.Tensor, n: int, bin_distance: torch.Tensor) -> torch.Tensor:
    """Algorithm 13: globularity loss for the H+L sub-complex.

    dgram_ab : (1, n, n, 128) — binder-only distogram block
    """
    probs = torch.softmax(dgram_ab, dim=-1)
    bd = bin_distance.clamp(max=27.0)
    e_sq = (probs * bd.pow(2)).sum(-1)   # (1, n, n)
    tril = torch.tril(torch.ones(n, n, device=dgram_ab.device), diagonal=-1)
    rg = ((e_sq[0] * tril).sum() / (n * n)).sqrt()
    return F.elu(rg - 2.38 * (n ** 0.365))


def _structural_loss(dlogits: torch.Tensor,
                      ab_mask: torch.Tensor, a_mask: torch.Tensor,
                      bin_distance: torch.Tensor) -> torch.Tensor:
    """Biohub weighted structural loss on 3-chain (H+L+A) distogram.

    dlogits    : (1, L, L, 128)
    ab_mask    : (L,) bool  — H+L positions
    a_mask     : (L,) bool  — antigen positions
    Returns scalar.
    """
    is_binder = ab_mask.float()
    is_target  = a_mask.float()
    n_ab = int(ab_mask.sum())

    intra = _contact_loss(dlogits[0], bin_distance, k=2, min_sep=9, cutoff=14.0,
                          chain_mask=is_binder, binder_mask=is_binder)
    inter = _contact_loss(dlogits[0], bin_distance, k=1, min_sep=0, cutoff=22.0,
                          chain_mask=is_target,  binder_mask=is_binder)
    # Binder sub-block: advanced indexing preserves gradient
    dgram_ab = dlogits[:, ab_mask, :, :][:, :, ab_mask, :]  # (1, n_ab, n_ab, 128)
    glob = _glob_loss(dgram_ab, n_ab, bin_distance)

    return 0.5 * intra + 0.5 * inter + 0.2 * glob


# ---- Model loading -----------------------------------------------------------

def build_model(device: str):
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel
    # ESMFold2-Experimental-Fast: 24 trunk layers, MSA disabled, 128-bin distogram.
    # Designed for gradient-based sequence optimization (vs 48-layer standard model).
    model = ESMFold2ExperimentalModel.from_pretrained("biohub/ESMFold2-Experimental-Fast").to(device).eval()
    return model, ESMFold2InputBuilder, ProteinInput, StructurePredictionInput


# ---- Score request -----------------------------------------------------------

def fold_and_score(model, Builder, ProteinInput, SPInput, heavy, light, antigen,
                   num_loops, num_steps, seed, device):
    spi = SPInput(sequences=[
        ProteinInput(id="H", sequence=heavy),
        ProteinInput(id="L", sequence=light),
        ProteinInput(id="A", sequence=antigen),
    ])
    with torch.no_grad():
        result = Builder().fold(model, spi,
                                num_loops=num_loops,
                                num_sampling_steps=num_steps,
                                num_diffusion_samples=1,
                                seed=seed)

    entity = result.entity_id.cpu().numpy()
    mask_H = entity == 0
    mask_L = entity == 1
    mask_A = entity == 2
    mask_ab = mask_H | mask_L

    pc = result.pair_chains_iptm.cpu().float().numpy()  # (3,3): H=0,L=1,A=2
    pae = result.pae.cpu().float().numpy()
    pae_ab_ag = pae[np.ix_(mask_ab, mask_A)]
    pae_ag_ab = pae[np.ix_(mask_A, mask_ab)]

    return {
        "ab_ag_iptm": float((pc[0, 2] + pc[1, 2]) / 2),
        "h_ag_iptm": float(pc[0, 2]),
        "l_ag_iptm": float(pc[1, 2]),
        "iptm": float(result.iptm),
        "ptm": float(result.ptm),
        "interface_pae": float((pae_ab_ag.mean() + pae_ag_ab.mean()) / 2),
        "interface_pae_norm": float((1 / (1 + (pae_ab_ag / 10) ** 2)).mean()),
        "mean_plddt": float(result.plddt.cpu().float().numpy().mean()),
    }


# ---- 2-chain scFv score request ---------------------------------------------

def fold_and_score_scfv(model, Builder, ProteinInput, SPInput, scfv, antigen,
                        num_loops, num_steps, seed, device):
    """2-chain (Ag + scFv) fold for final VH/VL codesign evaluation.

    Matches binder_design.py hero critic: global iptm from a single-chain
    scFv (VH-linker-VL) folded against the antigen.
    """
    spi = SPInput(sequences=[
        ProteinInput(id="A", sequence=antigen),
        ProteinInput(id="B", sequence=scfv),
    ])
    with torch.no_grad():
        result = Builder().fold(model, spi,
                                num_loops=num_loops,
                                num_sampling_steps=num_steps,
                                num_diffusion_samples=1,
                                seed=seed)
    return {
        "iptm": float(result.iptm),
        "ptm": float(result.ptm),
        "mean_plddt": float(result.plddt.cpu().float().numpy().mean()),
    }


# ---- Gradient request --------------------------------------------------------

def grad_and_score(model, Builder, ProteinInput, SPInput, heavy, light, antigen,
                   num_loops, seed, device):
    """Differentiable trunk pass to compute Biohub structural fitness gradient.

    Memory strategy
    ---------------
    ESMC backbone (6B params) is run first under no_grad to produce
    lm_hidden_states, then cache is flushed.  The gradient pass receives
    lm_hidden_states directly so ESMC does not accumulate activations.
    Backprop only flows through: inputs_embedder → z_init → folding_trunk → distogram_head.

    ESMFold2-Experimental-Fast has 24 trunk layers (vs 48 for standard model) and
    MSA disabled, so num_loops=1 (2 trunk passes) uses ~3.2 GB checkpoint storage
    rather than ~6.4 GB — comfortably fits in 80 GB with model weights.

    Returns
    -------
    grad_h : list of lists, shape (L_heavy, 33)
        Gradient of structural fitness w.r.t. heavy-chain residue-type logits.
    grad_l : list of lists, shape (L_light, 33)
        Gradient of structural fitness w.r.t. light-chain residue-type logits.
    structural_fitness : float
        Value of structural_fitness at the current sequence (higher is better).
        structural_fitness = −(0.5·intra_contact + 0.5·inter_contact + 0.2·glob)
    """
    spi = SPInput(sequences=[
        ProteinInput(id="H", sequence=heavy),
        ProteinInput(id="L", sequence=light),
        ProteinInput(id="A", sequence=antigen),
    ])

    features, _ = Builder().prepare_input(spi, seed=seed, device=device)

    entity_id = features["entity_id"][0]   # (L,)
    h_mask  = entity_id == 0
    l_mask  = entity_id == 1
    ab_mask = (entity_id == 0) | (entity_id == 1)
    a_mask  = entity_id == 2

    # Step 1: pre-compute ESMC hidden states under no_grad.
    # lm_hidden_states are always detached before entering the trunk, so the
    # gradient never flows through ESMC. Pre-computing avoids re-running the 6B
    # model during the gradient pass.
    with torch.no_grad():
        lm_hidden_states = model._compute_lm_hidden_states(
            input_ids=features["input_ids"],
            asym_id=features["asym_id"],
            residue_index=features["residue_index"],
            mol_type=features["mol_type"],
            token_mask=features["token_attention_mask"],
            lm_mask_pct=0.0,
        )

    if device == "cuda":
        torch.cuda.empty_cache()

    res_type = features["res_type"][0]  # (L,)
    res_type_oh = F.one_hot(res_type.long(), num_classes=33).float().unsqueeze(0)  # (1, L, 33)
    res_type_soft = res_type_oh.detach().requires_grad_(True)

    # Step 2: gradient pass.
    # Model parameters frozen at server startup so only res_type_soft.grad accumulates.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )
    with torch.enable_grad(), autocast_ctx:
        output = model(
            **features,
            res_type_soft=res_type_soft,
            lm_hidden_states=lm_hidden_states,
            num_loops=1,          # 2 trunk passes: valid gradient direction, fits in 80 GB
            num_sampling_steps=1,
            num_diffusion_samples=1,
            seed=seed,
        )

    dlogits = output["distogram_logits"].float()  # (1, L, L, 128)
    bin_distance = _get_mid_points(device)

    loss = _structural_loss(dlogits, ab_mask, a_mask, bin_distance)

    # Negate: CTMC sampler maximizes fitness; structural_loss is to be minimized.
    (-loss).backward()

    heavy_grad = res_type_soft.grad[0, h_mask].float().cpu().numpy()  # (L_H, 33)
    light_grad = res_type_soft.grad[0, l_mask].float().cpu().numpy()  # (L_L, 33)
    return heavy_grad.tolist(), light_grad.tolist(), float(-loss.detach().cpu())


# ---- Server loop -------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fold_server] Loading ESMFold2-Experimental-Fast on {device}...", file=sys.stderr, flush=True)
    model, Builder, ProteinInput, SPInput = build_model(device)
    # Freeze all model parameters: scoring uses torch.no_grad() anyway, and gradient
    # requests only need res_type_soft.grad, not parameter gradients.  Freezing here
    # ensures gradient-checkpoint recomputation sees the same requires_grad state.
    model.requires_grad_(False)
    print("[fold_server] Model loaded.", file=sys.stderr, flush=True)

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_type = req.get("type", "score")

            if device == "cuda":
                torch.cuda.empty_cache()

            if req_type == "gradient":
                grad_h, grad_l, fitness = grad_and_score(
                    model, Builder, ProteinInput, SPInput,
                    heavy=req["heavy"],
                    light=req["light"],
                    antigen=req["antigen"],
                    num_loops=req.get("num_loops", 10),
                    seed=req.get("seed", 0),
                    device=device,
                )
                print(json.dumps({"grad_h": grad_h, "grad_l": grad_l, "structural_fitness": fitness}), flush=True)

            elif req_type == "score_scfv":
                scores = fold_and_score_scfv(
                    model, Builder, ProteinInput, SPInput,
                    scfv=req["scfv"],
                    antigen=req["antigen"],
                    num_loops=req.get("num_loops", 3),
                    num_steps=req.get("num_steps", 200),
                    seed=req.get("seed", 0),
                    device=device,
                )
                print(json.dumps(scores), flush=True)

            else:  # "score" (default) — 3-chain H+L+Ag
                scores = fold_and_score(
                    model, Builder, ProteinInput, SPInput,
                    heavy=req["heavy"],
                    light=req["light"],
                    antigen=req["antigen"],
                    num_loops=req.get("num_loops", 10),
                    num_steps=req.get("num_steps", 50),
                    seed=req.get("seed", 0),
                    device=device,
                )
                print(json.dumps(scores), flush=True)

        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
