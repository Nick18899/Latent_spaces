"""
Reusable noise robustness experiment for the NMR -> SMILES transformer.

Design goals:
- no architecture changes
- works on top of the existing batch structure from dataset.py/model.py
- adds mixed noise to all input characteristics
- supports layerwise analysis for carbon encoder, hydrogen encoder, and decoder
- computes empirical quantiles for Tanimoto / valid rate / layerwise L2 shift

Expected batch structure:
    batch['smiles']   : LongTensor [B, T]       # tokenized target smiles from collate_fn
    batch['C_NMR']    : dict with keys spectrum/intensity/multiplicity
    batch['H_NMR']    : dict with keys spectrum/intensity/multiplicity

The code decodes target SMILES directly from batch['smiles'], so no dataset rewrite is needed.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception as e:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    _RDKIT_IMPORT_ERROR = e
else:
    _RDKIT_IMPORT_ERROR = None


@dataclass
class NoiseConfig:
    alpha_grid: Sequence[float]
    repeats: int = 20
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    max_tokens: int = 200
    max_batches: Optional[int] = None
    seed: int = 42

    # Gaussian noise scales for continuous channels
    c_shift_scale: float = 1.0
    h_shift_scale: float = 0.05
    c_intensity_scale: float = 0.10
    h_intensity_scale: float = 0.10

    # categorical corruption for multiplicity
    mult_base_prob: float = 0.25
    mult_max_prob: float = 0.50

    # fingerprints for Tanimoto
    fp_radius: int = 2
    fp_nbits: int = 2048


# =========================
# small utils
# =========================

def require_rdkit() -> None:
    if Chem is None or DataStructs is None or AllChem is None:
        raise ImportError(
            "RDKit is required for valid SMILES and Tanimoto metrics. "
            f"Original import error: {_RDKIT_IMPORT_ERROR}"
        )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def clone_tensor_dict(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in d.items()}


def get_system_token_ids(tokenizer) -> set:
    return {
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.pad_id(),
        tokenizer.unk_id(),
    }


def decode_target_smiles_from_batch(batch: Dict[str, Any], tokenizer) -> List[str]:
    seqs = batch["smiles"]
    if isinstance(seqs, torch.Tensor):
        seqs = seqs.detach().cpu().tolist()

    system_tokens = get_system_token_ids(tokenizer)
    filtered = []
    for seq in seqs:
        filtered.append([tok for tok in seq if tok not in system_tokens])

    decoded = tokenizer.decode(filtered)
    if isinstance(decoded, str):
        decoded = [decoded]
    return list(decoded)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    require_rdkit()
    return Chem.MolFromSmiles(smiles) is not None


def tanimoto_similarity(smiles_true: str, smiles_pred: str, radius: int = 2, nbits: int = 2048) -> float:
    require_rdkit()
    mol_true = Chem.MolFromSmiles(smiles_true)
    mol_pred = Chem.MolFromSmiles(smiles_pred)
    if mol_true is None or mol_pred is None:
        return 0.0

    fp_true = AllChem.GetMorganFingerprintAsBitVect(mol_true, radius, nBits=nbits)
    fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius, nBits=nbits)
    return float(DataStructs.TanimotoSimilarity(fp_true, fp_pred))


def exact_match_canonical(smiles_true: str, smiles_pred: str) -> int:
    c_true = canonicalize_smiles(smiles_true)
    c_pred = canonicalize_smiles(smiles_pred)
    if c_true is None or c_pred is None:
        return 0
    return int(c_true == c_pred)


def first_eos_mask(idxs: torch.Tensor, eos_id: int) -> torch.Tensor:
    bsz, seq_len = idxs.shape
    mask = torch.zeros_like(idxs, dtype=torch.bool)
    eos_hits = idxs.eq(eos_id)
    for b in range(bsz):
        eos_pos = torch.where(eos_hits[b])[0]
        if len(eos_pos) == 0:
            mask[b, :] = True
        else:
            mask[b, : eos_pos[0].item() + 1] = True
    return mask


def pool_sequence(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    hidden: [B, T, D]
    mask: [B, T] bool
    returns pooled [B, D]
    """
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (hidden * mask_f).sum(dim=1) / denom


# =========================
# external forward with layerwise states
# =========================

@torch.inference_mode()
def forward_with_layerwise_states(
    model,
    src_c_nmr: Dict[str, torch.Tensor],
    src_h_nmr: Dict[str, torch.Tensor],
    tgt: torch.Tensor,
) -> Dict[str, Any]:
    """
    Runs the existing model modules step by step, without changing architecture,
    and returns layerwise hidden states for:
        - carbon encoder
        - hydrogen encoder
        - decoder
    """
    src_carbon_mask, src_hydrogen_mask, tgt_mask = model.generate_mask(
        src_carbon_spectrum=src_c_nmr["spectrum"],
        src_hydrogen_spectrum=src_h_nmr["spectrum"],
        tgt=tgt,
    )

    src_c_nmr_embedded = model.dropout(
        model.positional_encoding(model.spectrum_embedding(src_c_nmr, spectrum_type="C_NMR"))
    )
    src_h_nmr_embedded = model.dropout(
        model.positional_encoding(model.spectrum_embedding(src_h_nmr, spectrum_type="H_NMR"))
    )
    tgt_embedded = model.dropout(model.positional_encoding(model.decoder_embedding(tgt)))

    carbon_states: List[torch.Tensor] = []
    hydrogen_states: List[torch.Tensor] = []
    decoder_states: List[torch.Tensor] = []

    enc_carbon_output = src_c_nmr_embedded
    for enc_layer in model.encoder_carbon_layers:
        enc_carbon_output = enc_layer(enc_carbon_output, src_carbon_mask)
        carbon_states.append(enc_carbon_output)

    enc_hydrogen_output = src_h_nmr_embedded
    for enc_layer in model.encoder_hydrogen_layers:
        enc_hydrogen_output = enc_layer(enc_hydrogen_output, src_hydrogen_mask)
        hydrogen_states.append(enc_hydrogen_output)

    dec_output = tgt_embedded
    for dec_layer in model.decoder_layers:
        dec_output = dec_layer(
            x=dec_output,
            enc_carbon_output=enc_carbon_output,
            enc_hydrogen_output=enc_hydrogen_output,
            src_carbon_mask=src_carbon_mask,
            src_hydrogen_mask=src_hydrogen_mask,
            tgt_mask=tgt_mask,
        )
        decoder_states.append(dec_output)

    logits = model.fc(dec_output)
    return {
        "logits": logits,
        "carbon_states": carbon_states,
        "hydrogen_states": hydrogen_states,
        "decoder_states": decoder_states,
        "src_carbon_mask": src_carbon_mask,
        "src_hydrogen_mask": src_hydrogen_mask,
        "tgt_mask": tgt_mask,
    }


# =========================
# generation and layerwise pooling
# =========================

@torch.inference_mode()
def generate_smiles_and_layerwise_vectors(
    model,
    src_c_nmr: Dict[str, torch.Tensor],
    src_h_nmr: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    max_tokens: int = 200,
) -> Tuple[List[str], Dict[str, List[torch.Tensor]]]:
    """
    Greedy generation + pooled layerwise vectors.

    Returns:
        smiles_list: list[str]
        pooled: dict with keys carbon_encoder / hydrogen_encoder / decoder,
                each value is list of [B, D] tensors, one per layer.
    """
    model.eval()
    src_c_nmr = move_to_device(src_c_nmr, device)
    src_h_nmr = move_to_device(src_h_nmr, device)

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    system_tokens = get_system_token_ids(tokenizer)

    batch_size = src_c_nmr["spectrum"].shape[0]
    idxs = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    while idxs.shape[1] < max_tokens:
        logits = model(src_c_nmr=src_c_nmr, src_h_nmr=src_h_nmr, tgt=idxs)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idxs = torch.cat([idxs, next_tok], dim=1)
        if torch.all(torch.any(idxs == eos_id, dim=1)):
            break

    filtered = []
    for seq in idxs.detach().cpu().tolist():
        filtered.append([tok for tok in seq if tok not in system_tokens])
    smiles_list = tokenizer.decode(filtered)
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    outputs = forward_with_layerwise_states(
        model=model,
        src_c_nmr=src_c_nmr,
        src_h_nmr=src_h_nmr,
        tgt=idxs,
    )

    # masks to pool encoder / decoder sequences
    carbon_seq_mask = src_c_nmr["spectrum"].ge(0)
    hydrogen_seq_mask = src_h_nmr["spectrum"].ge(0)
    decoder_seq_mask = first_eos_mask(idxs, eos_id=eos_id)

    pooled = {
        "carbon_encoder": [pool_sequence(h, carbon_seq_mask) for h in outputs["carbon_states"]],
        "hydrogen_encoder": [pool_sequence(h, hydrogen_seq_mask) for h in outputs["hydrogen_states"]],
        "decoder": [pool_sequence(h, decoder_seq_mask) for h in outputs["decoder_states"]],
    }
    return list(smiles_list), pooled


# =========================
# noise
# =========================

def corrupt_multiplicity_tensor(
    multiplicity: torch.Tensor,
    alpha: float,
    base_prob: float,
    max_prob: float,
    sys_token_idx: int = 0,
    num_classes: int = 10,
) -> torch.Tensor:
    out = multiplicity.clone()
    p = min(base_prob * alpha, max_prob)
    if p <= 0:
        return out

    non_sys = out.ne(sys_token_idx)
    change_mask = (torch.rand_like(out.float()) < p) & non_sys
    if not change_mask.any():
        return out

    random_vals = torch.randint(
        low=1,
        high=num_classes,
        size=out.shape,
        dtype=out.dtype,
        device=out.device,
    )
    same = change_mask & random_vals.eq(out)
    if same.any():
        random_vals[same] = ((random_vals[same] - 1 + 1) % (num_classes - 1)) + 1

    out[change_mask] = random_vals[change_mask]
    return out


def add_noise_to_channel(
    channel: Dict[str, torch.Tensor],
    alpha: float,
    shift_scale: float,
    intensity_scale: float,
    mult_base_prob: float,
    mult_max_prob: float,
    num_mult_classes: int = 10,
) -> Dict[str, torch.Tensor]:
    out = clone_tensor_dict(channel)

    spectrum = out["spectrum"]
    intensity = out["intensity"]
    multiplicity = out["multiplicity"]

    valid_mask = spectrum >= 0
    if alpha <= 0:
        return out

    eps_s = torch.randn_like(spectrum)
    spectrum_noisy = spectrum + alpha * shift_scale * eps_s
    spectrum_noisy = torch.where(valid_mask, spectrum_noisy, spectrum)

    eps_i = torch.randn_like(intensity)
    intensity_noisy = intensity + alpha * intensity_scale * eps_i
    intensity_noisy = torch.clamp(intensity_noisy, min=0.0)
    intensity_noisy = torch.where(valid_mask, intensity_noisy, intensity)

    multiplicity_noisy = corrupt_multiplicity_tensor(
        multiplicity=multiplicity,
        alpha=alpha,
        base_prob=mult_base_prob,
        max_prob=mult_max_prob,
        sys_token_idx=0,
        num_classes=num_mult_classes,
    )

    out["spectrum"] = spectrum_noisy
    out["intensity"] = intensity_noisy
    out["multiplicity"] = multiplicity_noisy
    return out


def add_noise_to_batch(batch: Dict[str, Any], alpha: float, cfg: NoiseConfig, noise_mode: str = "all") -> Dict[str, Any]:
    noisy = copy.deepcopy(batch)

    c_shift_scale = cfg.c_shift_scale
    h_shift_scale = cfg.h_shift_scale
    c_intensity_scale = cfg.c_intensity_scale
    h_intensity_scale = cfg.h_intensity_scale
    mult_base_prob = cfg.mult_base_prob

    if noise_mode == "shift_only":
        c_intensity_scale = 0.0
        h_intensity_scale = 0.0
        mult_base_prob = 0.0
    elif noise_mode == "intensity_only":
        c_shift_scale = 0.0
        h_shift_scale = 0.0
        mult_base_prob = 0.0
    elif noise_mode == "multiplicity_only":
        c_shift_scale = 0.0
        h_shift_scale = 0.0
        c_intensity_scale = 0.0
        h_intensity_scale = 0.0
    elif noise_mode == "all":
        pass
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    noisy["C_NMR"] = add_noise_to_channel(
        batch["C_NMR"],
        alpha=alpha,
        shift_scale=c_shift_scale,
        intensity_scale=c_intensity_scale,
        mult_base_prob=mult_base_prob,
        mult_max_prob=cfg.mult_max_prob,
        num_mult_classes=10,
    )
    noisy["H_NMR"] = add_noise_to_channel(
        batch["H_NMR"],
        alpha=alpha,
        shift_scale=h_shift_scale,
        intensity_scale=h_intensity_scale,
        mult_base_prob=mult_base_prob,
        mult_max_prob=cfg.mult_max_prob,
        num_mult_classes=10,
    )
    return noisy


# =========================
# evaluation
# =========================

@torch.inference_mode()
def evaluate_one_batch_under_noise(
    model,
    batch: Dict[str, Any],
    tokenizer,
    device: torch.device,
    alpha: float,
    cfg: NoiseConfig,
    noise_mode: str = "all",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        sample_df: one row per sample with task metrics
        layerwise_df: one row per sample x component x layer with layerwise L2 shift
    """
    target_smiles = decode_target_smiles_from_batch(batch, tokenizer)

    pred_clean, pooled_clean = generate_smiles_and_layerwise_vectors(
        model=model,
        src_c_nmr=batch["C_NMR"],
        src_h_nmr=batch["H_NMR"],
        tokenizer=tokenizer,
        device=device,
        max_tokens=cfg.max_tokens,
    )

    noisy_batch = add_noise_to_batch(batch, alpha=alpha, cfg=cfg, noise_mode=noise_mode)
    pred_noisy, pooled_noisy = generate_smiles_and_layerwise_vectors(
        model=model,
        src_c_nmr=noisy_batch["C_NMR"],
        src_h_nmr=noisy_batch["H_NMR"],
        tokenizer=tokenizer,
        device=device,
        max_tokens=cfg.max_tokens,
    )

    sample_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []

    final_decoder_l2 = torch.norm(
        pooled_noisy["decoder"][-1] - pooled_clean["decoder"][-1], dim=1
    ).detach().cpu().numpy()

    for i, (y_true, y_clean, y_pred) in enumerate(zip(target_smiles, pred_clean, pred_noisy)):
        sample_rows.append(
            {
                "sample_idx": i,
                "alpha": float(alpha),
                "noise_mode": noise_mode,
                "true_smiles": y_true,
                "pred_clean": y_clean,
                "pred_noisy": y_pred,
                "valid": int(is_valid_smiles(y_pred)),
                "exact_match": int(exact_match_canonical(y_true, y_pred)),
                "tanimoto": float(tanimoto_similarity(y_true, y_pred, radius=cfg.fp_radius, nbits=cfg.fp_nbits)),
                "latent_l2": float(final_decoder_l2[i]),
            }
        )

    for component in ("carbon_encoder", "hydrogen_encoder", "decoder"):
        clean_layers = pooled_clean[component]
        noisy_layers = pooled_noisy[component]
        for layer_idx, (clean_vec, noisy_vec) in enumerate(zip(clean_layers, noisy_layers)):
            l2_vals = torch.norm(noisy_vec - clean_vec, dim=1).detach().cpu().numpy()
            for i, l2_val in enumerate(l2_vals):
                layer_rows.append(
                    {
                        "sample_idx": i,
                        "alpha": float(alpha),
                        "noise_mode": noise_mode,
                        "component": component,
                        "layer_idx": int(layer_idx),
                        "layer_l2": float(l2_val),
                    }
                )

    return pd.DataFrame(sample_rows), pd.DataFrame(layer_rows)


def run_noise_quantile_experiment(
    model,
    dataloader: Iterable[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    cfg: NoiseConfig,
    noise_modes: Sequence[str] = ("all",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        sample_long_df
        layerwise_long_df
    """
    seed_everything(cfg.seed)
    require_rdkit()
    model.eval()

    sample_parts: List[pd.DataFrame] = []
    layer_parts: List[pd.DataFrame] = []

    for batch_idx, batch in enumerate(dataloader):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break

        for noise_mode in noise_modes:
            for alpha in cfg.alpha_grid:
                for rep in range(cfg.repeats):
                    sample_df, layer_df = evaluate_one_batch_under_noise(
                        model=model,
                        batch=batch,
                        tokenizer=tokenizer,
                        device=device,
                        alpha=float(alpha),
                        cfg=cfg,
                        noise_mode=noise_mode,
                    )
                    sample_df["batch_idx"] = batch_idx
                    sample_df["rep"] = rep
                    layer_df["batch_idx"] = batch_idx
                    layer_df["rep"] = rep
                    sample_parts.append(sample_df)
                    layer_parts.append(layer_df)

    sample_columns = [
        "sample_idx", "alpha", "noise_mode", "true_smiles", "pred_clean", "pred_noisy",
        "valid", "exact_match", "tanimoto", "latent_l2", "batch_idx", "rep",
    ]
    layer_columns = [
        "sample_idx", "alpha", "noise_mode", "component", "layer_idx", "layer_l2", "batch_idx", "rep",
    ]

    sample_long_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame(columns=sample_columns)
    layerwise_long_df = pd.concat(layer_parts, ignore_index=True) if layer_parts else pd.DataFrame(columns=layer_columns)
    return sample_long_df, layerwise_long_df


# =========================
# summarization
# =========================

def summarize_quantiles(
    long_df: pd.DataFrame,
    metric: str,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    group_cols: Sequence[str] = ("noise_mode", "alpha"),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = long_df.groupby(list(group_cols))[metric]

    for group_key, values in grouped:
        values = values.dropna().to_numpy()
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        row["mean"] = float(np.mean(values)) if len(values) else np.nan
        for q in quantiles:
            row[f"q{int(q*100)}"] = float(np.quantile(values, q)) if len(values) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def summarize_binary_rate(
    long_df: pd.DataFrame,
    metric: str,
    group_cols: Sequence[str] = ("noise_mode", "alpha"),
) -> pd.DataFrame:
    return (
        long_df.groupby(list(group_cols))[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"{metric}_rate"})
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )


def summarize_layerwise_quantiles(
    layerwise_df: pd.DataFrame,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    group_cols: Sequence[str] = ("noise_mode", "component", "layer_idx", "alpha"),
) -> pd.DataFrame:
    return summarize_quantiles(
        long_df=layerwise_df,
        metric="layer_l2",
        quantiles=quantiles,
        group_cols=group_cols,
    )


# =========================
# plotting
# =========================

def plot_quantile_curves(
    summary_df: pd.DataFrame,
    metric_name: str,
    noise_mode: Optional[str] = None,
    component: Optional[str] = None,
    layer_idx: Optional[int] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 4),
):
    import matplotlib.pyplot as plt

    df = summary_df.copy()
    if noise_mode is not None and "noise_mode" in df.columns:
        df = df[df["noise_mode"] == noise_mode].copy()
    if component is not None and "component" in df.columns:
        df = df[df["component"] == component].copy()
    if layer_idx is not None and "layer_idx" in df.columns:
        df = df[df["layer_idx"] == layer_idx].copy()

    quantile_cols = sorted([c for c in df.columns if c.startswith("q")], key=lambda c: int(c[1:]))

    plt.figure(figsize=figsize)
    for col in quantile_cols:
        plt.plot(df["alpha"], df[col], marker="o", label=col)
    if "mean" in df.columns:
        plt.plot(df["alpha"], df["mean"], linestyle="--", label="mean")

    title_parts = [metric_name]
    if noise_mode is not None:
        title_parts.append(str(noise_mode))
    if component is not None:
        title_parts.append(str(component))
    if layer_idx is not None:
        title_parts.append(f"layer={layer_idx}")

    plt.xlabel("Noise strength α")
    plt.ylabel(ylabel or metric_name)
    plt.title(" | ".join(title_parts))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_binary_rate(
    summary_df: pd.DataFrame,
    metric_name: str,
    noise_mode: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 4),
):
    import matplotlib.pyplot as plt

    df = summary_df.copy()
    if noise_mode is not None and "noise_mode" in df.columns:
        df = df[df["noise_mode"] == noise_mode].copy()
    rate_col = [c for c in df.columns if c.endswith("_rate")][0]

    plt.figure(figsize=figsize)
    plt.plot(df["alpha"], df[rate_col], marker="o")
    plt.xlabel("Noise strength α")
    plt.ylabel(metric_name)
    title_suffix = f" ({noise_mode})" if noise_mode is not None else ""
    plt.title(f"{metric_name} vs noise{title_suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_quantile_scatter_style(
    long_df: pd.DataFrame,
    metric: str,
    noise_mode: Optional[str] = None,
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    figsize: Tuple[int, int] = (8, 6),
    point_size: float = 7.0,
    point_alpha: float = 0.85,
    show_points: bool = True,
    show_legend: bool = True,
    ylabel: Optional[str] = None,
):
    """
    Plot empirical quantile curves in a style similar to the user's example:
    light gray background, black scatter points, colored quantile curves, legend on the right.
    """
    import matplotlib.pyplot as plt

    df = long_df.copy()
    if noise_mode is not None and "noise_mode" in df.columns:
        df = df[df["noise_mode"] == noise_mode].copy()

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe columns.")

    if df.empty:
        raise ValueError("No rows left after filtering; nothing to plot.")

    # empirical quantiles by alpha
    rows = []
    grouped = df.groupby("alpha")[metric]
    for alpha, values in grouped:
        values = values.dropna().values
        row = {"alpha": float(alpha)}
        for q in quantiles:
            row[f"q{int(round(q * 100))}"] = float(np.quantile(values, q))
        rows.append(row)
    qdf = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#EBEBEB")
    fig.patch.set_facecolor("white")

    # grid similar to ggplot gray theme
    ax.grid(True, which="major", color="#BDBDBD", linewidth=1.0)
    ax.grid(True, which="minor", color="#D9D9D9", linewidth=0.8)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    if show_points:
        ax.scatter(df["alpha"], df[metric], s=point_size, c="black", alpha=point_alpha, linewidths=0, label="Werte")

    # color palette close to the example
    palette = {
        5: "#F0E400",
        10: "#C7B700",
        25: "#5DA53A",
        50: "#00A087",
        75: "#5DA53A",
        90: "#C7B700",
        95: "#F0E400",
    }

    for q in quantiles:
        qint = int(round(q * 100))
        col = f"q{qint}"
        color = palette.get(qint, None)
        ax.plot(
            qdf["alpha"],
            qdf[col],
            color=color,
            linewidth=2.0,
            label=fr"Q$_x$({qint}\%)",
        )

    ax.set_xlabel("Noise strength α")
    ax.set_ylabel(ylabel if ylabel is not None else metric)

    title = f"{metric}: empirical quantile curves"
    if noise_mode is not None:
        title += f" ({noise_mode})"
    ax.set_title(title)

    # clean spines for a softer look
    for spine in ax.spines.values():
        spine.set_color("#9E9E9E")

    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.show()

def make_layerwise_heatmap_data(
    layerwise_summary_df: pd.DataFrame,
    component: str,
    noise_mode: str = "all",
    value_col: str = "q50",
) -> pd.DataFrame:
    df = layerwise_summary_df.copy()
    df = df[(df["component"] == component) & (df["noise_mode"] == noise_mode)].copy()
    pivot = df.pivot(index="layer_idx", columns="alpha", values=value_col).sort_index()
    return pivot


def plot_layerwise_heatmap(
    layerwise_summary_df: pd.DataFrame,
    component: str,
    noise_mode: str = "all",
    value_col: str = "q50",
    figsize: Tuple[int, int] = (8, 4),
):
    import matplotlib.pyplot as plt

    heatmap_df = make_layerwise_heatmap_data(
        layerwise_summary_df=layerwise_summary_df,
        component=component,
        noise_mode=noise_mode,
        value_col=value_col,
    )

    plt.figure(figsize=figsize)
    plt.imshow(heatmap_df.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(heatmap_df.columns)), [str(c) for c in heatmap_df.columns], rotation=45)
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.xlabel("Noise strength α")
    plt.ylabel("Layer index")
    plt.title(f"Layerwise heatmap | {component} | {noise_mode} | {value_col}")
    plt.tight_layout()
    plt.show()


# =========================
# convenience
# =========================

def default_noise_config() -> NoiseConfig:
    return NoiseConfig(
        alpha_grid=[round(x, 2) for x in np.linspace(0.0, 0.40, 21)],
        repeats=20,
        quantiles=(0.1, 0.5, 0.9),
        max_tokens=200,
        max_batches=None,
        seed=42,
        c_shift_scale=1.0,
        h_shift_scale=0.05,
        c_intensity_scale=0.10,
        h_intensity_scale=0.10,
        mult_base_prob=0.25,
        mult_max_prob=0.50,
        fp_radius=2,
        fp_nbits=2048,
    )


EXAMPLE_USAGE = r'''
from noise_quantile_experiment import (
    default_noise_config,
    run_noise_quantile_experiment,
    summarize_quantiles,
    summarize_binary_rate,
    summarize_layerwise_quantiles,
    plot_quantile_curves,
    plot_binary_rate,
    plot_layerwise_heatmap,
)

cfg = default_noise_config()
# for a quick smoke test:
# cfg.alpha_grid = [0.0, 0.1, 0.2]
# cfg.repeats = 2
# cfg.max_batches = 2

sample_long_df, layerwise_long_df = run_noise_quantile_experiment(
    model=model,
    dataloader=test_loader,
    tokenizer=tokenizer,
    device=device,
    cfg=cfg,
    noise_modes=("all",),
)

# task-level summaries

tanimoto_summary = summarize_quantiles(sample_long_df, metric="tanimoto", quantiles=cfg.quantiles)
latent_summary = summarize_quantiles(sample_long_df, metric="latent_l2", quantiles=cfg.quantiles)
valid_summary = summarize_binary_rate(sample_long_df, metric="valid")

plot_quantile_curves(tanimoto_summary, metric_name="Tanimoto", noise_mode="all")
plot_quantile_curves(latent_summary, metric_name="Final decoder L2", noise_mode="all")
plot_binary_rate(valid_summary, metric_name="Valid rate", noise_mode="all")

# layerwise summaries
layerwise_summary = summarize_layerwise_quantiles(layerwise_long_df, quantiles=cfg.quantiles)
plot_quantile_curves(layerwise_summary, metric_name="Layer L2", noise_mode="all", component="decoder", layer_idx=0)
plot_layerwise_heatmap(layerwise_summary, component="decoder", noise_mode="all", value_col="q50")
plot_layerwise_heatmap(layerwise_summary, component="carbon_encoder", noise_mode="all", value_col="q50")
plot_layerwise_heatmap(layerwise_summary, component="hydrogen_encoder", noise_mode="all", value_col="q50")

sample_long_df.to_csv("noise_quantile_sample_long.csv", index=False)
layerwise_long_df.to_csv("noise_quantile_layerwise_long.csv", index=False)
layerwise_summary.to_csv("noise_quantile_layerwise_summary.csv", index=False)
'''
