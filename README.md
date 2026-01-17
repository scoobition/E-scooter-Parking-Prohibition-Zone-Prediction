# scoobition
Data-driven Decision Support for E-scooter Parking Management

---

## Project Overview

With the rapid expansion of shared electric scooter services in South Korea, the number of users has increased significantly. As a result, social conflicts related to improper parking of electric scooters have continued to grow. In particular, repeated towing incidents occur due to scooters obstructing pedestrian pathways or blocking building entrances.

Notably, some towing cases occur even in areas that are officially designated as permitted parking zones. This indicates that although a location may legally allow parking, it may not be suitable for electric scooter parking in real-world usage environments. However, current policies for defining and managing electric scooter return zones do not sufficiently reflect historical towing data or regional characteristics, leading to user inconvenience and repeated towing incidents.

This project analyzes historical electric scooter towing data to identify patterns of repeated towing in specific areas and predicts regions with a high likelihood of future towing incidents. The results are intended to provide data-driven evidence to support decision-making for adjusting electric scooter return zones and setting management priorities.

---

## Objectives

- Analyze historical electric scooter towing data to identify high-risk areas
- Predict future towing-prone regions based on spatial grid units
- Identify patterns of repeatedly towed areas using historical towing data
- Predict areas with a high probability of future towing incidents
- Support decision-making through: 
  - Grid-based risk visualization 
  - Top-K priority areas for management 
  - Analytical evidence for reviewing and adjusting parking (return) zones

This project does not aim to designate parking prohibition zones. Its purpose is to provide analytical evidence to support decisions made by local governments and service operators.

---

## Dataset

- **Source**: National Data Portal (Korea)
- **Scale**:
  - Monthly average: approximately 5,300–5,400 records
  - Total records: approximately 64,200
- **Time range**: January to December
- **Main attributes**:
  - Address
  - Towing occurrence date
  - Administrative district information
---

## Data Processing Pipeline

1. Load monthly raw CSV files
2. Extract and clean address data
3. Aggregate duplicate addresses and count occurrences
4. Perform geocoding only on unique addresses
5. Cache geocoding results for efficiency
6. Convert latitude/longitude into **200m × 200m spatial grids**
7. Assign unique grid IDs and aggregate counts per grid
8. Visualize grid-based towing distributions
9. Create lag-based features using the previous 3 months
10. Predict the next month’s towing counts per grid

---

## Modeling Approach

- **Problem type**: Regression-based time series prediction
- **Model**: Random Forest Regressor
- **Training strategy**:
  - Sliding window approach using the previous 3 months
  - Example:
    - (Jan, Feb, Mar) → Apr
    - (Feb, Mar, Apr) → May
- **Training period**: January–November
- **Prediction target**: December grid-level towing counts

Random Forest was selected due to its robustness to non-linear relationships, spatial heterogeneity, and relatively stable performance with limited temporal depth compared to ARIMA or LSTM-based models.

---
## Features

- Grid-based spatial visualization of towing incidents
- Monthly heatmaps using 200m spatial resolution
- Predicted vs. actual comparison maps for December
- Ranking of high-risk grids based on predicted towing frequency

---

## Tech Stack

- **Language**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Spatial Processing**: GeoPandas
- **Visualization**: Matplotlib, Folium
- **Geocoding**: Google Maps API

---

## Project Structure

```text
E-scooter-Parking-Prohibition-Zone-Prediction/
├─ src/
│  ├─ __pycache__/
│  ├─ google_geocode.py
│  ├─ grid.py
│  ├─ io_loader.py
│  ├─ make_features.py
│  ├─ pipeline_geo.py
│  ├─ predict_rf.py
│  ├─ preprocess.py
│  ├─ result.py
│  ├─ reverse_geocode_top10.py
│  ├─ train_rf.py
│  ├─ visualize_pred.py
│  └─ viz_grid_map.py
│
├─ data/
│  ├─ 12_result_grid_meta.csv
│  ├─ 12_result_predata.csv
│  ├─ 12_result.csv
│  ├─ after.csv
│  ├─ features.csv
│  ├─ geocode_cache_result.csv
│  ├─ geocode_cache.csv
│  ├─ grid_meta.csv
│  ├─ pred_12.csv
│  ├─ predata_12.csv
│  ├─ predata.csv
│  └─ top10_with_address.csv
│
├─ document/
│  └─ documents
│
├─ map/
│  └─ ( heatmap / pred / real HTML reault file by month)
│
├─ original_data/
│  ├─ 1.csv
│  ├─ 2.csv
│  ├─ 3.csv
│  ├─ 4.csv
│  ├─ 5.csv
│  ├─ 6.csv
│  ├─ 7.csv
│  ├─ 8.csv
│  ├─ 9.csv
│  ├─ 10.csv
│  ├─ 11.csv
│  └─ 12.csv
│
├─ demo_gui.py
├─ main.py
├─ model_rf.pkl
├─ procedure.txt
└─ README.md
```
---

## How to Run
1. Create a `.env` file in the same directory as `main.py` and add your API keys: 
- ```
  GOOGLE_MAPS_API_KEY=your_google_api_key
  JUSO_CONFM_KEY=your_juso_api_key
  ```
2. Run the project:
- Run in the terminal:
  ```
  python main.py
  ```
- Run with the GUI:
  ```
  python demo_gui.py
  ```


---

## Authors

KwanHo Kwon  
Computer Information Engineering  

SoYoung Lee  
Computer Information Engineering  
