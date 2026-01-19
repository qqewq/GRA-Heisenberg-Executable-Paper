# ============================================
# src/hilbert_space.py — Гильбертовы пространства
# ============================================

import torch
import numpy as np

class HilbertSpace:
    """Гильбертово пространство домена"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.basis = self._create_orthonormal_basis(dim)
    
    def _create_orthonormal_basis(self, dim: int) -> torch.Tensor:
        """Ортогональный базис пространства"""
        # Упрощённо: стандартный базис
        basis = torch.eye(dim, dtype=torch.complex64)
        return basis
    
    def project_to_goal(self, goal_states: torch.Tensor) -> torch.Tensor:
        """Проектор P_G0 на пространство решений цели"""
        # Упрощённая реализация: проекция на главные компоненты
        U, S, Vh = torch.linalg.svd(goal_states)
        P_goal = U @ torch.diag(S / (S + 1e-8)) @ U.T.conj()
        return P_goal[:self.dim, :self.dim]
