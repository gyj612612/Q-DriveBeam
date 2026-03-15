from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _consistency_loss(fused_logits: torch.Tensor, branch_logits: list[torch.Tensor]) -> torch.Tensor:
    p_log = F.log_softmax(fused_logits, dim=-1)
    p = F.softmax(fused_logits, dim=-1)
    total = torch.tensor(0.0, device=fused_logits.device)
    for b in branch_logits:
        q = F.softmax(b, dim=-1)
        q_log = F.log_softmax(b, dim=-1)
        total = total + 0.5 * (
            F.kl_div(p_log, q, reduction="batchmean") + F.kl_div(q_log, p, reduction="batchmean")
        )
    return total / max(1, len(branch_logits))


def compute_losses(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    branch_aux_lambda: float,
    consistency_lambda: float,
    gate_reg_lambda: float,
    iemf_enabled: bool = False,
    iemf_psai: float = 1.2,
    iemf_scale_min: float = 0.5,
    iemf_scale_max: float = 2.0,
    iemf_detach_coeff: bool = True,
    ae_recon_lambda: float = 0.0,
    ae_kl_lambda: float = 0.0,
) -> Dict[str, torch.Tensor]:
    fused_logits = output["fused_logits"]
    branch_logits = output["branch_logits"]
    gate_reg = output["gate_reg"]
    ae_rec = output.get("ae_rec_loss", fused_logits.new_tensor(0.0))
    ae_kl = output.get("ae_kl_loss", fused_logits.new_tensor(0.0))

    main_per = F.cross_entropy(fused_logits, labels, reduction="none")
    iemf_coeff = labels.new_ones(labels.size(0), dtype=fused_logits.dtype)

    if branch_logits and iemf_enabled:
        if iemf_detach_coeff:
            with torch.no_grad():
                p_f = F.softmax(fused_logits, dim=-1)
                idx = torch.arange(labels.size(0), device=labels.device)
                ratios = []
                for b in branch_logits:
                    p_b = F.softmax(b, dim=-1)
                    ratios.append(p_b[idx, labels] / (p_f[idx, labels] + 1e-6))
                ratio_mean = torch.stack(ratios, dim=0).mean(dim=0)
                iemf_coeff = 1.0 + torch.tanh(float(iemf_psai) * (1.0 - ratio_mean))
                iemf_coeff = iemf_coeff.clamp(min=float(iemf_scale_min), max=float(iemf_scale_max))
        else:
            p_f = F.softmax(fused_logits, dim=-1)
            idx = torch.arange(labels.size(0), device=labels.device)
            ratios = []
            for b in branch_logits:
                p_b = F.softmax(b, dim=-1)
                ratios.append(p_b[idx, labels] / (p_f[idx, labels] + 1e-6))
            ratio_mean = torch.stack(ratios, dim=0).mean(dim=0)
            iemf_coeff = 1.0 + torch.tanh(float(iemf_psai) * (1.0 - ratio_mean))
            iemf_coeff = iemf_coeff.clamp(min=float(iemf_scale_min), max=float(iemf_scale_max))

    if iemf_detach_coeff:
        iemf_coeff = iemf_coeff.detach()
    loss_main = torch.mean(main_per * iemf_coeff)

    if branch_logits:
        branch_per = sum(F.cross_entropy(x, labels, reduction="none") for x in branch_logits) / len(branch_logits)
        loss_branch = torch.mean(branch_per * iemf_coeff)
        loss_consistency = _consistency_loss(fused_logits, branch_logits)
    else:
        loss_branch = torch.tensor(0.0, device=labels.device)
        loss_consistency = torch.tensor(0.0, device=labels.device)

    total = (
        loss_main
        + branch_aux_lambda * loss_branch
        + consistency_lambda * loss_consistency
        + gate_reg_lambda * gate_reg
        + ae_recon_lambda * ae_rec
        + ae_kl_lambda * ae_kl
    )
    return {
        "total": total,
        "main": loss_main,
        "branch": loss_branch,
        "consistency": loss_consistency,
        "gate_reg": gate_reg,
        "ae_rec": ae_rec,
        "ae_kl": ae_kl,
        "iemf_coeff_mean": iemf_coeff.mean(),
    }
