"""Gradient-trained VQC: the same ZZ feature map + RealAmplitudes circuit as `vqc.py`, simulated exactly in
PyTorch so it can be trained with Adam (COBYLA struggles with 20-60 parameters).

Circuit (n qubits, L layers):
    ZZ feature map, reps=2, linear:  [H, P(2 x_i)] then for (i, i+1): CX, P(2 (pi - x_i)(pi - x_{i+1})), CX
    RealAmplitudes, reps=L, linear:  RY(theta) layer, then L x [CX chain, RY(theta) layer]
    readout: P(y=1) = P(odd parity of the measured bitstring)  (qiskit VQC's default parity interpretation)
"""
import math

import numpy as np
import torch
import torch.nn as nn


class _State:
    def __init__(self, batch, n):
        self.n = n
        psi = torch.zeros(batch, 2**n, dtype=torch.complex128)
        psi[:, 0] = 1
        self.psi = psi.reshape(batch, *([2] * n))

    def _ax(self, q):
        return self.n - q  # qubit 0 = least significant (Qiskit ordering)

    def u(self, U, q):
        ax = self._ax(q)
        p = self.psi.movedim(ax, -1)
        p = torch.einsum("...j,ij->...i", p, U) if U.dim() == 2 else torch.einsum("b...j,bij->b...i", p, U)
        self.psi = p.movedim(-1, ax)

    def cx(self, c, t):
        ca, ta = self._ax(c), self._ax(t)
        idx = [slice(None)] * self.psi.dim()
        idx[ca] = 1
        sub = self.psi[tuple(idx)]
        new = self.psi.clone()
        new[tuple(idx)] = sub.flip(ta - 1 if ta > ca else ta)
        self.psi = new


_H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)


def _p(phi):  # phase gate diag(1, e^{i phi}), batched
    one = torch.ones_like(phi, dtype=torch.complex128)
    zero = torch.zeros_like(one)
    e = torch.exp(1j * phi.to(torch.complex128))
    return torch.stack([torch.stack([one, zero], -1), torch.stack([zero, e], -1)], -2)


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)


class TorchVQC(nn.Module):
    def __init__(self, n_qubits, layers=6, fm_reps=2, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.L, self.fm_reps = n_qubits, layers, fm_reps
        self.theta = nn.Parameter(torch.rand(n_qubits * (layers + 1), generator=g, dtype=torch.float64) * 2 * math.pi)
        parity = torch.tensor([bin(i).count("1") % 2 for i in range(2**n_qubits)], dtype=torch.float64)
        self.register_buffer("parity", parity)

    def forward(self, x):
        x = x.to(torch.float64)
        s = _State(x.shape[0], self.n)
        for _ in range(self.fm_reps):
            for q in range(self.n):
                s.u(_H, q)
                s.u(_p(2 * x[:, q]), q)
            for q in range(self.n - 1):
                s.cx(q, q + 1)
                s.u(_p(2 * (math.pi - x[:, q]) * (math.pi - x[:, q + 1])), q + 1)
                s.cx(q, q + 1)
        k = 0
        for q in range(self.n):
            s.u(_ry(self.theta[k]), q); k += 1
        for _ in range(self.L):
            for q in range(self.n - 1):
                s.cx(q, q + 1)
            for q in range(self.n):
                s.u(_ry(self.theta[k]), q); k += 1
        probs = (s.psi.abs() ** 2).reshape(x.shape[0], -1)
        return probs @ self.parity  # P(y = 1)


class TorchVQCClassifier:
    def __init__(self, n_qubits, layers=6, epochs=200, lr=0.05, seed=42):
        self.n, self.layers, self.epochs, self.lr, self.seed = n_qubits, layers, epochs, lr, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.model = TorchVQC(self.n, self.layers, seed=self.seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64)
        self.loss_history = []
        for _ in range(self.epochs):
            opt.zero_grad()
            p = self.model(xt).clamp(1e-7, 1 - 1e-7)
            loss = nn.functional.binary_cross_entropy(p, yt)
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        p = self.model(torch.tensor(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def to_qiskit(self):
        """Trained circuit (feature map with free inputs + bound ansatz) for hardware runs."""
        from qiskit.circuit.library import real_amplitudes, zz_feature_map

        fm = zz_feature_map(self.n, reps=2, entanglement="linear")
        an = real_amplitudes(self.n, reps=self.layers, entanglement="linear")
        return fm.compose(an.assign_parameters(self.model.theta.detach().numpy()))
