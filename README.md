# Quantum Diabetes Prediction

An **optimized variational quantum classifier (VQC)** for diabetes prediction on the PIMA Indian Diabetes dataset,
comparing qubit counts, circuit depth and feature maps with classical classifiers.

## Method
1. **Pre-processing**: impossible zeros (glucose, blood pressure, skin thickness, insulin, BMI) → training-set median;
   standardise; for 4 qubits keep the 4 most informative features (ANOVA F-test); scale to [0, π].
2. **Feature map**: ZZ feature map (2 repetitions, linear entanglement), with Z and Pauli (Z, YY) maps for comparison.
3. **Ansatz**: RealAmplitudes with *L* layers (default 6), linear entanglement.
4. **Training**: COBYLA on cross-entropy, exact statevector simulation (qiskit-machine-learning `VQC`).
5. **Evaluation**: stratified 80 / 20 split; accuracy, precision, recall, F1, ROC-AUC vs. logistic regression,
   RBF-SVM and random forest.

## Repository Structure
```
Quantum-Diabetes-Prediction/
├── Code/
│   ├── data.py            # loading, imputation, feature selection, angle scaling
│   ├── vqc.py             # feature maps + RealAmplitudes VQC
│   ├── train.py           # quantum and classical experiments -> Results/results.json
│   ├── plot_results.py    # comparison, training curves, circuit diagrams
│   └── requirements.txt
├── Dataset/               # pima-indians-diabetes.csv + description
├── Figures/
├── Results/
├── LICENSE
└── README.md
```

## Quick Start
```bash
cd Code
pip install -r requirements.txt
python train.py                                        # 4 and 8 qubits, ZZ map, 6 layers
python train.py --qubits 8 --layers 1 2 4 6 --feature-maps zz z pauli
python plot_results.py
```

## Publication
**Diabetes Prediction Using an Optimized Variational Quantum Classifier**  
*International Journal of Intelligent Systems*, 2025. DOI: [10.1155/int/1351522](https://doi.org/10.1155/int/1351522)  
Reported: a VQC with six layers and a ZZ feature map reached **0.85 accuracy with 8 qubits** and **0.80 with 4 qubits**.

## Results
See [Results/README.md](Results/README.md).

## Author
**Wajiha Rahim Khan**  
[Google Scholar](https://scholar.google.com/citations?user=ctvOkbYAAAAJ) · [Email](mailto:wajihakhan906@gmail.com)

## License
MIT. See [LICENSE](LICENSE).
