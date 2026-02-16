# Repository: Multi-Hazard Risk Assessment for Maize Production in South Africa

This repository provides the companion code for the research article:

**Assessing Multi-Hazard Risk Dynamics in South African Maize Production: A Framework Integrating Hazard Interactions, Temporal and Spatial Variability, and Irrigation Contexts.**

## **Overview**
Climate change amplifies agricultural risks, particularly for **rainfed maize production in South Africa**. This repository contains the scripts used in the study to analyze the **spatio-temporal dynamics of climate-induced hazards**, their interactions, and the role of irrigation in mitigating risks. 

The study introduces a **multi-hazard risk score system**, integrating:
- **City-level clustering** based on irrigation coverage.
- **Climatic hazard profiling** (drought, heatwaves, cold spells, excessive rainfall, and disease risk).
- **Multi-hazard interaction analysis** linking risk scores to yield variability.
- **OLS and Random Forest models** to predict yield based on climatic conditions and hazard interactions.

The repository provides all necessary scripts to replicate data preprocessing, clustering, risk assessment, and statistical modeling.

## **Folder Structure**
```
📂 project_root/
├── 📂 data/                # Raw and processed data - not included in version control
├── 📂 scripts/             # Python scripts for modeling and analysis (after refactoring)
│   ├── 📂 bak/             # Original non-refactored script files
├── 📂 results/             # Output results (plots, model outputs, metrics)  - not included in version control
│   ├── plots/
│   ├── metrics/
├── 📜 requirements.txt     # Dependencies required to run the code
├── 📜 .gitignore           # Ignore large or sensitive files (e.g., data, logs)
├── 📜 README.md            # Documentation and instructions for use
```

## **Installation & Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/Spark-Towers/multi-hazard-agricultural-climate-risk.git
   cd multi-hazard-agricultural-climate-risk
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Running Scripts**
Each script accepts command-line arguments for flexible execution.

### **Example Usage**
#### **OLS Model**
```bash
python scripts/ols_models.py --file data/input.xlsx --cluster_ref C1 --cv_splits 5
```
#### **Random Forest Model**
```bash
python scripts/random_forest_model.py --file data/input.xlsx --n_estimators 100
```

## **Dependencies** (`requirements.txt`)
```
pandas
numpy
matplotlib
seaborn
statsmodels
sklearn
scipy
patsy
```

## **.gitignore**
```
# Ignore data files
/data/
/results/
*.csv
*.xlsx
*.log
*.json

# Ignore virtual environment
venv/
__pycache__/
*.pyc
```

## **Contributors**
- **Sophie Grosse** *(sophie.grosse@sparktowers.com)*
- **Mohammade Check** *(mohammade.check@sparktowers.com)*

## **Research Contribution**
This repository provides a reproducible analytical framework to:
- Identify structural and territorial patterns of vulnerability across South African maize farming systems.
- Quantify synergies and trade-offs between sensitivity and adaptive capacity indicators.
- Assess how different vulnerability dimensions interact across natural, social, human, physical, and financial capitals.
- Detect statistically significant relationships using Pearson correlations with multiple-testing corrections (FDR and Bonferroni).
- Explore structural relationships through Principal Component Analysis (PCA) at city and typology levels.
- Generate spatially interpretable visual outputs (radar glyphs, heatmaps, biplots) to support territorial comparison.

The code enables researchers and policymakers to analyze multi-dimensional vulnerability structures, identify reinforcing or counteracting mechanisms across capitals, and support evidence-based climate adaptation strategies in agricultural systems.

## **Authors & Citation**
If you use this repository in your research, please cite:
> **S. Grosse, T. Berchoux, M. Check, H. Belhouchette, N. Baghdadi (2026).** Synergies and trade-offs in vulnerability to climate hazard: a structural and territorial analysis of South African maize farming systems. [Journal Name].
