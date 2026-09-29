# Quantum Diabetes Prediction

An **optimized variational quantum classifier (VQC)** for diabetes prediction on the PIMA Indian Diabetes dataset,
comparing qubit counts, circuit depth and feature maps with classical classifiers.

![Study workflow](Figures/paper/workflow.png)

## Method
1. **Pre-processing**: impossible zeros (glucose, blood pressure, skin thickness, insulin, BMI) → training-set median;
   standardise; scale to [0, π]. **8 qubits** use all 8 attributes; **4 qubits** use the attributes selected in the
   study (**Glucose, Age, Insulin, BMI**), or an ANOVA F-test selection with `--attributes select`.
2. **Feature map**: ZZ feature map (2 repetitions, linear entanglement), with Z and Pauli (Z, YY) maps for comparison.
3. **Ansatz**: RealAmplitudes with *L* layers (default 6), linear entanglement.
4. **Training**: COBYLA on cross-entropy, exact statevector simulation (qiskit-machine-learning `VQC`).
5. **Evaluation**: stratified 80 / 20 split; accuracy, precision, recall, F1, ROC-AUC vs. logistic regression,
   RBF-SVM and random forest.

## Published Study Design
As shown in the workflow above, the paper evaluates:
1. **18 variants**: three feature maps × three optimizers × two attribute sets (8 and 4 qubits);
2. **54 variants**: 2, 4 and 6 layers with 24, 40 and 54 trainable parameters;
3. **Quantum error mitigation (QEM)** with scaling factors λ ∈ {1, 0.2, 0.1, 0.01};
4. **Six classical models** (LR, DT, GB, XGBoost, RF, SVM) on both attribute sets for comparison.

## Published Figures
The best model, a **6-layer VQC with QEM and λ = 0.01 on eight attributes**, gives the highest accuracy and recall of
all models compared:

![VQC vs classical models](Figures/paper/vqc_vs_classical_8_attributes.png)

Effect of quantum error mitigation and the scaling factor λ on accuracy over training iterations:

![QEM scaling](Figures/paper/qem_scaling_accuracy.png)

Pairwise distributions of the eight PIMA attributes by outcome:

![PIMA pair plot](Figures/paper/pima_pairplot.png)

*Figures reproduced from Khan et al., "Diabetes Prediction Using an Optimized Variational Quantum Classifier",
International Journal of Intelligent Systems, 2025, [doi:10.1155/int/1351522](https://doi.org/10.1155/int/1351522),
published open access under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.*

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
│   └── paper/             # figures from the published article (CC BY 4.0)
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
