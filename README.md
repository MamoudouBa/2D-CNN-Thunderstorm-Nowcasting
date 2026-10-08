2-D CNN and U-Net Models for Probabilistic Thunderstorm Nowcasting
This repository contains the official code, preprocessing pipelines, model architectures, and evaluation scripts for short-term (1 - 8 h) probabilistic thunderstorm nowcasting over the Contiguous United States (CONUS).
The models ingest operational High-Resolution Rapid Refresh (HRRR) Numerical Weather Prediction (NWP) fields to predict the probability of thunderstorm occurrence (defined by radar composite reflectivity > 30/35 dBZ) mapped against Multi-Radar/Multi-Sensor (MRMS) observations.
CONUS Spatial Domain & Grid Configuration
•	Spatial Coverage: Contiguous United States (25o  - N 50oN, 120oW -72oW)
•	Grid Resolution: 0.1otimes 0.1o ~ 10 km horizontal resolution)
•	Target Leads: Forecast lead times from f01 to f08 (1 - 8 hours)
Predictor Variables
Input features are extracted directly from operational HRRR model initialization grids and standardized via zero-mean, unit-variance scaling (StandardScaler):
Thermodynamic & Moisture Predictors
•	Precipitable Water: Total Precipitable Water (mm)
•	Relative Humidity: 850-hPa Relative Humidity (%)
•	Surface Instability: Surface-Based Convective Available Potential Energy (CAPE, J kg-1)
•	Convective Inhibition: Convective Inhibition (CIN, J kg-1)
•	Lifted Index: Lifted Index (LI, K)
•	Precipitation: Convective Precipitation Accumulation (kg m-2)
Kinematic & Dynamics Predictors
•	Surface Wind Components:U and V zonal/meridional surface wind components (m s-1)
•	Low-Level Wind Components: 850-hPa, U and V zonal/meridional wind components (m s-1)
•	Vertical Velocity: Vertical velocity at 925 hPa, 700 hPa, and 500 hPa (Pa s-1)
•	Vertical Wind Shear:
o	Surface (10 m) to 800-hPa layer wind shear (times 10-3 s-1)
o	800-hPa to 600-hPa layer wind shear (10-3 s-1)
Deep Learning Architectures
1.	2-D CNN: A non-pooled 5-layer convolutional network utilizing 5 times 5 spatial kernels, SpatialDropout2D regularization, and weighted categorical cross-entropy loss.
2.	U-Net: An encoder-decoder architecture with skip connections, evaluated under Z-score standardization and early stopping (ModelCheckpoint) to achieve optimal feature convergence across mesoscale boundaries.
Repository Contents
•	train_2d_cnn.py – Model definition and training pipeline with dynamic learning rate decay (ReduceLROnPlateau).
•	evaluate_models.py – Brier Score, Reliability Diagram, and Critical Success Index (CSI) verification utilities.
•	preprocess_data.py – Z-score feature scaling and multi-year temporal dataset splitting utilities.
•	weights/ – Pre-trained network weights for the 2-D CNN and U-Net models.
Citation & Contact
For technical questions or bug reports, please open an issue in this repository or contact Mamoudou Ba (mamoudou.ba@noaa.gov).

