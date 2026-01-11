# scoobition
Data-driven Decision Support for E-scooter Parking Management

---

## Project Overview

With the rapid expansion of shared electric scooter services in South Korea, the number of users has increased significantly. As a result, social conflicts related to improper parking of electric scooters have continued to grow. In particular, repeated towing incidents occur due to scooters obstructing pedestrian pathways or blocking building entrances.

Notably, some towing cases occur even in areas that are officially designated as permitted parking zones. This indicates that although a location may legally allow parking, it may not be suitable for electric scooter parking in real-world usage environments. However, current policies for defining and managing electric scooter return zones do not sufficiently reflect historical towing data or regional characteristics, leading to user inconvenience and repeated towing incidents.

This project analyzes historical electric scooter towing data to identify patterns of repeated towing in specific areas and predicts regions with a high likelihood of future towing incidents. The results are intended to provide data-driven evidence to support decision-making for adjusting electric scooter return zones and setting management priorities.

---

## Objectives

- Identify patterns of repeatedly towed areas using historical towing data
- Predict areas with a high probability of future towing incidents
- Support decision-making through:
  - Top-K priority areas for management
  - Candidate areas for reviewing and adjusting permitted parking (return) zones

This project does not aim to designate parking prohibition zones. Its purpose is to provide analytical evidence to support decisions made by local governments and service operators.

---

## Features

1. Visual analysis of electric scooter towing status in Seoul
2. Visualization of towing pattern analysis results
3. Visualization of predicted towing-prone areas
4. Ranking of areas based on towing frequency and risk

---

## Dataset

- Seoul Metropolitan Government electric scooter towing records
- Data includes district information, address, and towing type

---

## Tech Stack

Python  
NumPy  
Pandas  
Matplotlib  
Seaborn  
Folium  
scikit-learn  
HDBSCAN / DBSCAN  
GeoPandas  

---

## Project Structure


---

## How to Run


---

## Authors

KwanHo Kwon  
Computer Information Engineering  

SoYoung Lee  
Computer Information Engineering  
