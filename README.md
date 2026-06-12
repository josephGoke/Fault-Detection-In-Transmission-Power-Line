# Fault Detection in Transmission Power Line

This project focuses on the detection and classification of electrical faults in power transmission lines using Deep Learning techniques. Currently, it leverages a **Variational Autoencoder (VAE)** for representation learning and anomaly detection, with plans to integrate a **supervised classification model** on top of the latent features for precise fault type categorization.

## Project Description

Power transmission lines are susceptible to various types of faults (Line-to-Ground, Line-to-Line, etc.) which can disrupt grid stability. This project aims to:

- Detect the occurrence of a fault using current and voltage measurements.
- Classify the specific type of fault (A, B, C phases and Ground).
- Utilize unsupervised learning (VAE) to capture complex patterns in power system data.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Workflow](#workflow)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Architecture](#model-architecture)
7. [Future Work](#future-work)

## Dataset Description

The project uses datasets containing simulated power system measurements:

- **`classData.csv`**: Contains features for fault classification.
- **`detect_dataset.csv`**: Focused on binary fault detection.

### Data Columns

- **Fault Indicators**: `G` (Ground), `A`, `B`, `C` (Phases).
- **Current Readings**: `Ia`, `Ib`, `Ic`.
- **Voltage Readings**: `Va`, `Vb`, `Vc`.

## Workflow

1. **Data Preprocessing**: Loading and cleaning the Kaggle dataset, handling distributions, and feature scaling.
2. **Exploratory Data Analysis (EDA)**: Visualizing fault patterns, phase distributions, and correlation between voltage/current shifts during faults.
3. **Representation Learning**: Training a Variational Autoencoder (VAE) to compress the high-dimensional electrical data into a latent space.
4. **Classification (In Progress)**: Building a supervised layer on top of the VAE's latent representations to categorize specific fault types.

## Features

- **Deep Learning based**: Uses PyTorch for implementing the VAE.
- **Comprehensive EDA**: Detailed visualization of phase currents and voltages under different fault conditions.
- **Hybrid Approach**: Combining unsupervised feature extraction with supervised classification.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/josephGoke/Fault-Detection-In-Transmission-Power-Line.git
   cd research
   ```

2. Install dependencies:

   ```bash
   pip install torch pandas numpy matplotlib seaborn
   ```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook fault-detection.ipynb
   ```

2. Run the cells to perform EDA and train the VAE model.
3. Use `VAE.py` to customize the autoencoder architecture.

## Model Architecture

### Variational Autoencoder (VAE)

The current implementation in `VAE.py` includes:

- **Encoder**: Compresses input (10 dimensions) into a latent space.
- **Latent Space**: Uses `mu` and `logvar` for the reparameterization trick.
- **Decoder**: Reconstructs the electrical signals from the latent vector.

## Future Work

- [ ] Complete the integration of the **Supervised Classification Model** on top of the VAE encoder.
- [ ] Evaluate performance against traditional ML models (Random Forest, XGBoost).
- [ ] Implement real-time fault detection stream processing.

---
*Note: This project is currently in active development. The primary focus is moving from anomaly detection to full fault classification.*
