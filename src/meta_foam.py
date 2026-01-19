# ============================================
# src/meta_foam.py — Вычисление Φ_meta
# ============================================

import torch

def compute_meta_foam(psi_states: list, projectors: list) -> float:
    """Φ_meta(Ψ_ens, G_tot) — полная междоменная пена"""
    M = len(psi_states)
    phi_meta = 0.0
    
    # P_tot = усреднённый проектор
    P_tot = sum(P for P in projectors) / len(projectors)
    
    for i in range(M):
        for j in range(i+1, M):
            psi_i = psi_states[i].view(-1, 1)
            psi_j = psi_states[j].view(-1, 1)
            # |<Ψ^(i)|P_tot|Ψ^(j)>|^2
            off_diag = torch.abs(psi_i.T.conj() @ P_tot @ psi_j).item() ** 2
            phi_meta += off_diag
    
    return phi_meta

def gradient_meta_foam(psi_states: list, projectors: list, 
                      target_idx: int) -> torch.Tensor:
    """∂Φ_meta/∂Ψ^(a) — мета-градиент"""
    P_tot = sum(P for P in projectors) / len(projectors)
    grad = torch.zeros_like(psi_states[target_idx])
    
    M = len(psi_states)
    for i in range(M):
        if i == target_idx: continue
        psi_i = psi_states[i].view(-1, 1)
        psi_target = psi_states[target_idx].view(-1, 1)
        
        # ∂/∂Ψ^(a) |<Ψ^(i)|P_tot|Ψ^(a)>|^2
        term = 2 * torch.real(psi_i.T.conj() @ P_tot @ psi_target)
        grad += torch.real(term).squeeze()
    
    return grad
