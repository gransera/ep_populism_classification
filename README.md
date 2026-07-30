# Populism Beyond the State
### A Transformer-Based Longitudinal Analysis of Populist Rhetoric in the European Parliament

**Antonia Granser** | Master's Thesis | Geschwister-Scholl-Institut für Politikwissenschaft, LMU Munich

---

## Overview

This project develops and applies a transformer-based classification pipeline to detect populist rhetoric in European Parliament debates across five legislative terms (1999–2024). It combines dictionary-based methods, fine-tuned RoBERTa classifiers, and panel regression models to analyse how anti-elitism and people-centrism vary across populist party groups and over time.

---

## Repository Structure

'''
├── data/ # Not included (see Data Sources below)
├── notebooks/ # Jupyter notebooks for classification, analysis, and visualisation
├── outputs/
│ ├── regression/ # Regression tables and coefficient plots
│ ├── graphs/ # Time series and descriptive visualisations
│ ├── bar_charts/ # Party-period bar charts per dimension
│ ├── tables/ # Party-period tables per dimension & dictionary-transformer comparison
│ └── metrics/ # Metrics of transformer-based approach
└── README.md
'''

---

## Data Sources

The raw data is not included in this repository due to size. It can be accessed via the following sources:

**EP Speech Corpus (ParlLaw)**
Full corpus of European Parliament debates:
https://search.gesis.org/research_data/SDN-10.7802-2824

**Populist Parties in the European Parliament**
Dataset on the participation of populist national parties in the EP:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RFRCZS

**RoBERTa Language Model**
Base model used for fine-tuning the populism classifiers:
https://huggingface.co/FacebookAI/roberta-large

---

## Methods

The project operationalises populism along two core dimensions:

- **Anti-Elitism** — rhetoric targeting corrupt or self-serving elites
- **People-Centrism** — rhetoric appealing to the virtuous people as a homogeneous group

For classification a novel transformer-based multilabel classifier is trained for each populist dimension:

 **Transformer-based classification** — fine-tuned RoBERTa-large models, trained separately for each dimension

Results are aggregated to the party-year level and analysed using panel OLS regression models with party fixed effects and Driscoll-Kraay standard errors.

---

## Requirements & Installation

```bash
conda env create -f environment.yml
conda activate ma_ep_populism
```

---

## How to Run

Run the notebooks in the following order:

1. **Training** — fine-tuning the RoBERTa-large models for anti-elitism and people-centrism classification
2. **Model Application** — dictionary matching and transformer inference on the full EP speech corpus
3. **Validation** — comparison of dictionary vs. transformer results against human annotations
4. **Descriptive Analysis** — exploratory analysis of populist rhetoric across parties, groups, and time
5. **Aggregation** — sentence-level results aggregated to party × year × dimension
6. **Regression** — panel models estimating drivers of populist rhetoric
7. **Visualisation** — descriptive plots and regression coefficient visualisations

---

## Outputs

- Around 4 million classified sentences according to the populism dimensions
- Normalised time series of populist rhetoric per legislative year
- Party-period bar charts per populist dimension
- Validation metrics (precision, recall, F1) for classification approaches
- Regression tables and coefficient forest plots
- Significance heatmaps and R² comparisons across model types

---

## Citation

If you use this code or build on this work, please cite:

> Granser, A. (2025). *Populism Beyond the State: A Transformer-Based Longitudinal Analysis of Populist Rhetoric in the European Parliament*. Master's Thesis, Geschwister-Scholl-Institut für Politikwissenschaft, LMU Munich.


