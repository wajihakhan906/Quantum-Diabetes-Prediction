# Dataset: PIMA Indian Diabetes

`pima-indians-diabetes.csv`: 768 female patients of Pima Indian heritage (age ≥ 21), from the National Institute of
Diabetes and Digestive and Kidney Diseases, via the UCI Machine Learning Repository. No header row.

| Column | Description |
|---|---|
| Pregnancies | number of pregnancies |
| Glucose | plasma glucose, 2 h oral glucose tolerance test (mg/dL) |
| BloodPressure | diastolic blood pressure (mm Hg) |
| SkinThickness | triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/mL) |
| BMI | body-mass index (kg/m²) |
| DiabetesPedigreeFunction | family-history score |
| Age | years |
| Outcome | 1 = diabetes within 5 years, 0 = not (268 positive / 500 negative) |

Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI denote missing values; `Code/data.py` replaces them with
the training-set median. Split: stratified 80 / 20 train / test (seed 42).
