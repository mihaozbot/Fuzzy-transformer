# 📈 Fuzzformer: A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting

This repository contains the implementation of **Fuzzformer**, a novel hybrid neural architecture that combines deep sequence modeling with interpretable fuzzy inference for **long-term multivariate stock market forecasting**.

The core model integrates:
- **LSTM** for temporal encoding of financial sequences  
- **Multi-Head Self-Attention** for highlighting relevant time steps  
- **Gaussian-based fuzzy rules** for interpretability  
- **ARIX local models** for multi-horizon forecasting
- 
## 🗂 Repository Structure

```
.
├── LSTM/              # Baseline LSTM forecasting model
├── Transformer/       # Transformer-based deep clustering experiments
├── Neuro-fuzzy/       # Core fuzzy logic, clustering, and ARIX implementation
├── evolvingsystem/    # Final Fuzzformer training and architecture scripts
├── Prophet/           # Prophet forecasting baseline
├── utils/             # Data loading, preprocessing, visualization
├── models/            # Model definitions and checkpoints
├── matlab/            # Related MATLAB simulation code
├── data/              # Financial time series (S&P 500, VIX, etc.)
├── requirements.txt   # Python package dependencies
├── Evolver__An_Evolving_Neuro_Fuzzy_System_for_Interpretable_Long_Term_Stock_Market_Forecasting.pdf
```


## 📊 Dataset

The model was trained and evaluated on a multivariate dataset including:

- **S&P 500 index** (main forecast target)
- **VIX Volatility Index**
- **Gold prices**
- **5-Year U.S. Treasury Yield**

| Time Period | 2001–2023 |
|-------------|-----------|
| Split       | 80% Train / 10% Validation / 10% Test |
| Scaling     | Min-Max normalization |


### Run the Fuzzformer model

The main experiment is implemented in:

```bash
neuro_fuzzy_experiment.ipynb
```

## 🎯 Key Features

- ✅ Hybrid LSTM + Transformer + Fuzzy Inference model
- ✅ Interpretable Gaussian-based clustering in latent space
- ✅ Winner-takes-all rule optimization for clearer rule boundaries
- ✅ Integrates classic ARIX models for stable long-term prediction
- ✅ Regularized deep clustering using Bhattacharyya and KL losses


## 📄 Paper

> **Title**: *A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting*  
> **Status**: Under Review  
> 📎 PDF: [See included paper](./Evolver__An_Evolving_Neuro_Fuzzy_System_for_Interpretable_Long_Term_Stock_Market_Forecasting.pdf)

## ✍️ Authors

- **Miha Ožbot** — University of Ljubljana  
- **Igor Škrjanc** — University of Ljubljana  
- **Vitomir Štruc** — University of Ljubljana  

## 📜 License

This repository is provided for academic and research use. Licensing will be updated after the review process concludes.
