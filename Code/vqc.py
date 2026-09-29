"""Variational quantum classifier: ZZ feature map + RealAmplitudes ansatz (qiskit-machine-learning)."""
from qiskit.circuit.library import pauli_feature_map, real_amplitudes, z_feature_map, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

FEATURE_MAPS = {
    "zz": lambda n: zz_feature_map(n, reps=2, entanglement="linear"),
    "z": lambda n: z_feature_map(n, reps=2),
    "pauli": lambda n: pauli_feature_map(n, reps=2, paulis=["Z", "YY"]),
}


def build_vqc(n_qubits, layers=6, feature_map="zz", maxiter=150, seed=42, callback=None):
    fm = FEATURE_MAPS[feature_map](n_qubits)
    ansatz = real_amplitudes(n_qubits, reps=layers, entanglement="linear")
    return VQC(feature_map=fm, ansatz=ansatz, optimizer=COBYLA(maxiter=maxiter),
               sampler=StatevectorSampler(seed=seed), callback=callback)
