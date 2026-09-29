"""Checks the PyTorch VQC against Qiskit's statevector for random inputs and weights."""
import numpy as np
import torch
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.quantum_info import Statevector

from vqc_torch import TorchVQC


def test_matches_qiskit():
    for n, L in [(4, 2), (4, 6), (8, 6)]:
        m = TorchVQC(n, L, seed=1)
        X = np.random.default_rng(0).uniform(0, np.pi, (3, n))
        ours = m(torch.tensor(X)).detach().numpy()
        fm = zz_feature_map(n, reps=2, entanglement="linear")
        an = real_amplitudes(n, reps=L, entanglement="linear").assign_parameters(m.theta.detach().numpy())
        for b in range(3):
            probs = Statevector(fm.assign_parameters(X[b]).compose(an)).probabilities()
            ref = sum(p for i, p in enumerate(probs) if bin(i).count("1") % 2)
            assert abs(ours[b] - ref) < 1e-9, (n, L, ours[b], ref)
