#  Forecasting Hourly Energy Consumption with LSTM Networks

This project focuses on predicting **hourly power usage** using Long Short-Term Memory (LSTM) neural networks to improve **energy grid efficiency** and enable proactive grid management. By leveraging the PJM Interconnection dataset, we aim to develop a robust deep learning model that captures complex time-series patterns in electricity consumption data.

---

##  Problem Statement

Accurate forecasting of hourly energy consumption is vital for efficient electricity distribution, cost savings, and sustainability. Traditional models often fail to capture the non-linear and dynamic nature of power usage patterns caused by:

- Time of day and day of week
- Weather conditions
- Public holidays
- Human activity trends

---

##  Objectives

- Preprocess PJM energy dataset (handle missing values, outliers, and scaling)
- Engineer temporal features (hour, day, weekend/weekday, holidays)
- Visualize consumption patterns and correlation insights
- Develop an optimized LSTM-based forecasting model
- Evaluate model performance using RMSE, MAE, and MAPE
- Compare results with baseline models (Persistence, ARIMA)
- Deploy an accurate and interpretable forecasting solution

---

##  Technologies Used

| Tool/Library     | Purpose                                  |
|------------------|-------------------------------------------|
| **Python**       | Core programming language                 |
| **Pandas**       | Data manipulation and preprocessing       |
| **NumPy**        | Numerical computations                    |
| **Matplotlib / Seaborn** | Data visualization              |
| **TensorFlow / Keras** | LSTM model building                |
| **scikit-learn** | Evaluation metrics and scaling            |
| **holidays**     | Generate national/public holiday features |

---

##  Dataset

**PJM Hourly Energy Consumption Dataset**  
- Covers power usage across 13 U.S. states and Washington, D.C.
- Hourly consumption records from **2002 to present**
- Publicly available and granular to regional levels

---

##  Implementation Overview

1. **Data Preprocessing**
   - Imputation for missing values
   - Outlier detection (IQR/Z-score)
   - Min-Max scaling for normalization

2. **Feature Engineering**
   - Time-based (hour, day, month, holidays)
   - Lagged and rolling statistical features
   - External signals (e.g., weather/temperature if available)

3. **Modeling with LSTM**
   - LSTM layers with dropout for generalization
   - Optimized using grid search for hyperparameters
   - Early stopping to avoid overfitting

4. **Evaluation**
   - Compared against ARIMA and Persistence models
   - Performance metrics: RMSE, MAE, MAPE

---

##  Results & Key Findings

- **15% improvement** in forecast accuracy vs ARIMA baseline
- Effective prediction of **peak demand periods**
- LSTM effectively learns **daily and weekly trends**
- Training and validation loss converged consistently
- Framework applicable for real-time grid forecasting

---

##  Future Scope

- Incorporate real-time **weather forecasts**
- Use advanced deep learning variants (e.g., Bi-LSTM, Transformers)
- Expand to **renewable energy forecasting**
- Deploy as a live dashboard or web app for grid operators

---

##  Contributors

- **Soumil Mukhopadhyay** (16010122257)  
- **Sharwar Patil** (16010122278)  
- **Shreyas Nair** (16010122274)  
- **Rohit Sharan** (16010122307)

---

##  Presentation

The full project presentation is available in the [`Presentation/`](./Presentation/) folder:  
📂 [`Forecasting-Hourly-Energy-Consumption-with-LSTM-Nw.pptx`](./Presentation/Forecasting-Hourly-Energy-Consumption-with-LSTM-Nw.pptx)

---

##  Summary

This project demonstrates how deep learning can significantly improve electricity consumption forecasting. LSTM networks offer a data-driven approach to predict usage trends, helping pave the way for **smart grids**, **load balancing**, and **energy sustainability**.

---

