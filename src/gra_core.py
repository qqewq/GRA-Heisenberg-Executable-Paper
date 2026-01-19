# ============================================
# src/gra_core.py — GRA Мета-обнулёнка (ядро)
# ============================================

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import torch  # PyTorch для градиентов

@dataclass
class DomainGoal:
    """Цель домена"""
    name: str
    projector: torch.Tensor  # P_G0^(a) — проектор цели
    hilbert_dim: int

class GRAMetaNulling:
    """GRA Мета-обнулёнка — основная архитектура"""
    
    def __init__(self, domains: List[DomainGoal], lambda_meta: float = 1.0):
        self.domains = domains
        self.lambda_meta = lambda_meta
        self.psi_states = []  # Ψ^(a) для каждого домена
        self.M = len(domains)
        
        # Инициализация состояний
        for domain in domains:
            psi = torch.randn(domain.hilbert_dim, dtype=torch.complex64)
            psi = psi / torch.norm(psi)  # Нормировка
            self.psi_states.append(psi)
    
    def local_foam(self, a: int, psi_a: torch.Tensor) -> float:
        """Φ^(a)(Ψ^(a), G₀^(a)) — локальная пена разума"""
        P = self.domains[a].projector
        psi = psi_a.view(-1, 1)
        foam = torch.abs((psi.T.conj() @ P @ psi).real - torch.abs(psi.T.conj() @ psi).real)**2
        return foam.item()
    
    def meta_foam(self) -> float:
        """Φ_meta(Ψ_ens, G_tot) — междоменная пена"""
        total_foam = 0.0
        # G_tot = среднее по всем проекторам (упрощённо)
        P_tot = torch.zeros_like(self.domains[0].projector)
        for domain in self.domains:
            P_tot += domain.projector
        P_tot = P_tot / len(self.domains)
        
        for i in range(self.M):
            for j in range(i+1, self.M):
                psi_i = self.psi_states[i].view(-1, 1)
                psi_j = self.psi_states[j].view(-1, 1)
                cross = torch.abs(psi_i.T.conj() @ P_tot @ psi_j).item()**2
                total_foam += cross
        return total_foam
    
    def J_meta(self) -> float:
        """Полный мета-функционал J_meta(Ψ)"""
        local_sum = sum(self.local_foam(a, self.psi_states[a]) for a in range(self.M))
        meta = self.lambda_meta * self.meta_foam()
        return local_sum + meta
    
    def optimize(self, epochs: int = 1000, lr: float = 0.01) -> Dict[str, Any]:
        """Оптимизация: ∂Ψ/∂t = -∇_Ψ J_meta"""
        history = {'J': [], 'local_foams': [], 'meta_foam': []}
        
        optimizer = torch.optim.Adam([psi.requires_grad_() for psi in self.psi_states], lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            J = self.J_meta()
            
            J.backward()
            optimizer.step()
            
            # Нормировка состояний
            for i, psi in enumerate(self.psi_states):
                norm = torch.norm(psi)
                if norm > 0:
                    self.psi_states[i] = psi / norm
            
            history['J'].append(J.item())
            history['local_foams'].append([self.local_foam(a, self.psi_states[a]) 
                                         for a in range(self.M)])
            history['meta_foam'].append(self.meta_foam())
        
        return history
