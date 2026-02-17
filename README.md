# KLOE ML Analysis

Machine learning approach for signal-background discrimination in KLOE experiment data.

## 📋 Project Overview

This project applies machine learning (XGBoost) to improve signal-background separation in particle physics data from the KLOE experiment. The ultimate goals include:

- Enhancing signal purity by reducing combinatorial and physical backgrounds
- Enabling searches for rare signals
- Improving understanding of detector response

## 🚀 Current Status

- ✅ Simplified Monte Carlo generator implemented
- ✅ Toy XGBoost model developed and tested
- ⏳ Real KLOE data access pending approval
- ⏳ Full physics description in preparation

## 📊 Methodology

### Data Generation
Currently using a simplified MC generator that simulates:
- Signal events: [describe your signal process]
- Background events: [describe background sources]

*Note: Real KLOE data will be incorporated once project approval is obtained.*

### Machine Learning Approach
- **Algorithm**: XGBoost classifier
- **Features**: [list key features used]
- **Training/Test split**: 80/20
- **Evaluation metrics**: ROC AUC, signal efficiency, background rejection

## 🛠️ Installation

```bash
git clone https://github.com/[username]/kloe-ml-analysis.git
cd kloe-ml-analysis
pip install -r requirements.txt