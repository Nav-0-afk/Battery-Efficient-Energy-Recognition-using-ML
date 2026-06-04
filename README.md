# 🔋 Energy-Aware Human Activity Recognition (HAR) for Edge Inference

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-Hardware_Target-lightgrey.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://fake-license-link.com/MIT)

> **Academic Context:** This repository contains the Machine Learning end-semester project developed for the B.Tech Artificial Intelligence and Engineering program at Amrita Vishwa Vidyapeetham, Bangalore. 

Continuous Human Activity Recognition (HAR) on wearable devices imposes significant energy demands. Standard machine learning implementations often fail in the real world due to **tensor emulation bloat** when compiled via edge frameworks like ONNX. This bloat causes execution latencies that exceed strict 10 ms hardware deadlines (100 Hz sensor sampling), forcing continuous CPU utilization and rapidly draining the battery.

This project solves this bottleneck by replacing standard deep learning models with a **structurally pruned, cascaded routing architecture** evaluated on the UCI HAR Dataset. By actively deflecting simple data and filtering noise, the system drastically reduces active CPU time, enabling a **>90% deep-sleep duty cycle** while maintaining **94% clinical accuracy**.

### Dataset Requirement
This project requires the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). 
Download and extract the zip file into a folder named `dataset/` in the root directory before running the pre-processing scripts.

---

## ✨ Key Features

* **Sub-Millisecond ONNX Execution:** Maximum pipeline execution path completes in **~963 µs**, comfortably beating the 10 ms hardware deadline.
* **Cascaded Routing Funnel:** Uses mathematically lightweight gatekeepers to shield computationally heavy multi-class tree ensembles from unnecessary executions.
* **Strict Structural Pruning:** Models are bounded mathematically (e.g., XGBoost `max_depth=4`, Isolation Forest `max_samples=16`) to survive edge compiler bloat.
* **RF-RFE Dimensionality Reduction:** Optimal kinetic feature space reduced from 561 to 200 essential variables using RandomizedSearchCV tuning.
* **Interactive Dashboard:** Includes a Streamlit UI to visualize latency, accuracy, and hardware feasibility metrics.

---

## 🧠 Architectural Pipeline

The system routes real-time sensor data through four sequential gates, ensuring early computational exits whenever possible:

1. **Stage 0: Anomaly Gate (Isolation Forest)**
   * **Role:** Unsupervised filter to block non-activity sensor noise.
   * **Optimization:** Pruned to 30 trees to cap mathematical depth, preventing ONNX emulation latency spikes.
2. **Stage 1: Gatekeeper (Logistic Regression)**
   * **Role:** Separates low-entropy static activities from high-entropy dynamic activities.
   * **Optimization:** Pure mathematical dot product; introduces virtually zero computational drag (~15 µs in ONNX).
3. **Stage 1B: Static Specialist (Linear SVC)**
   * **Role:** Resolves static postures (e.g., Sitting, Standing, Lying).
   * **Optimization:** Maximum margin hyperplanes provide absolute classification confidence for rapid CPU sleep triggers.
4. **Stage 2: Dynamic Specialist (XGBoost)**
   * **Role:** Classifies high-entropy kinetic movements (e.g., Walking, Running, Climbing).
   * **Optimization:** Capped at 80 estimators to guarantee deterministic, level-wise tree traversal on the MCU instruction cache.

---

## 📂 Repository Structure

```text
├── artifacts/                     # Scalers, selected feature indices, and preprocessed numpy arrays
├── compilers/                     
│   └── exporting_to_onnx.py       # Scripts for ONNX graph translation
├── dataset/                       
│   └── UCI HAR Dataset/           # Raw source dataset
├── models/                        # Trained baseline and alternative models (.joblib, .json, .txt)
├── onnx_models/                   # Compiled .onnx models for edge deployment latency testing
├── alternative_architecture.ipynb # EDA and testing of abandoned models (Decision Trees, LightGBM, etc.)
├── app.py                         # Streamlit dashboard for hardware feasibility visualization
├── benchmarking_onnx.py           # Latency measurement scripts for the ONNX runtime
├── feature_selection.py           # RF-RFE implementation and RandomizedSearchCV
├── final_architecture.ipynb       # Notebook defining the optimized cascaded pipeline
├── inference.py                   # End-to-end inference routing script
├── initial_architecture.ipynb     # Baseline unoptimized models
├── pre_processing.py              # Data scaling and categorical encoding
└── train_models.py                # Main training pipeline
```
---

## 🔮 Future Roadmap (Hardware Engineering)
While ONNX compilation successfully meets the current real-time deadlines, future iterations will target ultra-low power execution:

1. **Bare-Metal C-Code Translation:** Utilizing m2cgen to extract raw .joblib weights and translate the pipeline into zero-dependency C-code (pure if-else branches), bypassing ONNX emulation entirely.

3. **RTOS Integration:** Deploying the C-code payload into a FreeRTOS environment on an ARM Cortex-M4 MCU, executing DMA sensor I/O and inference on separate hardware threads.

4. **Physical Sensor Toggling:** Wiring the MCU to issue SPI commands to physically cut power to high-drain peripherals (like the gyroscope) when Stage 1 detects a static state.

> [!NOTE]
> This architecture is grounded in energy-aware machine learning strategies codified in recent HAR literature (Contoli et al., 2024).
