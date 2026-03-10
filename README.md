# 🧠 Topology Optimisation: Mesh-Invariant Upscaling using Fourier Neural Operators

<div align="center">
  <em>Accelerating structural engineering algorithms while preserving strict adherence to physical laws.</em>
</div>
<br>

This repository provides a Python-based framework for 2D **Topology Optimisation (TopOpt)**, featuring classical mathematical solvers alongside a **deep learning pipeline** leveraging **Fourier Neural Operators (FNOs)**. 

The primary objective is to drastically accelerate the generation of optimal multi-material geometries while circumventing the grid-resolution limitations of traditional computer vision models.

---
## � Codebase Structure

The repository is modularized into core mathematical solvers and deep learning models:

```text
├── models/                     # Deep Learning Architecture
│   ├── fno.py                  # Fourier Neural Operator (FNO) Definition
│   └── <model_name>/           # Auto-generated workspaces for train runs (config, weights)
├── src/                        # Core Mathematical Solvers (Ground Truth)
│   ├── top99_2d.py             # 2D Single-Material (SIMP) Solver
│   ├── multitop_2d.py          # 2D Multi-Material (Alternating Phase) Solver
│   ├── FEM_models.py           # Shared FEM & Filter logic (Assembly, Stiffness)
│   └── visualisation.py        # Plotting utilities for 2D structures
├── assets/                     # Images and benchmark visualisations
├── train_model.py              # Central script to train neural operators
├── eval_model.py               # Evaluation script for metrics & inference
└── requirements.txt            # Python dependencies
```

## 🏗️ What is Topology Optimisation?

At its core, Topology Optimisation is a mathematical method that distributes material within a given design space, subject to applied loads and boundary conditions and a target volume fraction of the material. The goal is to maximize the performance of a structure (e.g., maximizing stiffness). 

Traditionally, algorithms iteratively "carve" and solve computationally expensive Finite Element Method (FEM) equations. As grid resolutions scale, these traditional algorithms suffer from profound computational bottlenecks.

---

---

## 📉 The Roadblock: Image-Based Upscaling (Thesis Limitations)

My initial [master's thesis research](https://hal.science/hal-03717882/document) sought to replace the expensive classical solvers by treating optimal geometries as standard 2D images, training Generative Adversarial Networks (GANs) to predict higher-resolution structural outcomes from low-resolution approximations.

**The Limitations:**
Conventional Convolutions (CNNs/GANs) are fundamentally tied to discrete pixel grids. Despite generating outputs with reasonable *perceptual* geometry, the structures suffered from:
1. **Severe noise** (Incoherent Material Distribution).
2. **Failure to Generalise** (models when trained on composite datasets containing solutions from two boundary conditions ( i.e. cantilevers and Supported Beams) failed to provide descernable solution).

<div align="center">
  <img src="assets/experiment1%20outputs.png" alt="Exp Upscaling Cantilevers on 40x20 grid to 80x40" width="700"/>
  <br>
  <em>SR-GAN outputs: Visually structured, but mechanically flawed.</em>
</div>
<div align="center">
  <img src="assets/experiment3%20outputs.png" alt="Exp Upscaling Cantilevers and Simply Supported Beam on 40x20 grid to 80x40" width="700"/>
  <br>
  <em>SR-GAN outputs: Visually structured, but mechanically flawed.</em>
</div>

---

## 🚀 The Solution: Fourier Neural Operators (FNO)

To overcome the problems of pixel-bound networks, this codebase shifts the paradigm to **Operator Learning**.

Instead of learning a mapping between fixed pixel matrices, **Fourier Neural Operators** learn to map between infinite-dimensional continuous function spaces. 

### Why FNOs for Structural Integrity?
- **True Mesh-Invariance:** By evaluating weights in the continuous Fourier domain, the network inherently understands the PDE. We can train the architecture on a computationally cheap low-resolution mesh (e.g., `40x20`) and directly evaluate it on a massive high-resolution mesh (e.g., `100x50`) **zero-shot**, natively circumventing the scaling issues of standard GANs.
- **Noise Remediation:** Truncating high-frequency noise modes during the Fast Fourier Transform mathematically forces the network to learn smooth, structurally contiguous solutions.

---

## 💻 FNO Upscaling Pipeline

The repository ships with an end-to-end automated FNO training and evaluation framework specifically adapted to `nelx × nely` (width × height) FEM coordinate conventions. 

### 1. Training (`train_model.py`)
Provides an automated pipeline for paired geometry learning:
- **Auto-Splitting:** Automatically matches input and target datasets, partitioning them blindly into a strict `80/20` train-test division to ensure valid metrics.
- **Model Workspaces:** Passing `--model_name my_fno` isolates checkpoints, loss curves, and generates a dynamic `config.json` containing the precise neural architecture.

```bash
# Example: Train a base FNO model
python train_model.py \
    --model_name fno_base \
    --data_in_dir ./CNT_40x20/CNT_40x20 \
    --data_out_dir ./CNT_80X40/CNT_80X40 \
    --data_val_dir ./CNT_100x50/CNT_100x50 \
    --batch_size 8 \
    --epochs 60
```

### 2. Evaluation & Inference (`eval_model.py`)
Dynamically reinstantiates models based on their configuration signature.

- **Batch Benchmarking:** Evaluate the final model against the unseen 20% validation split, computing strict pixel-wise Mean Squared Errors (MSE) against continuous predictions.
- **Zero-Shot Inference:** Provide a singular low-resolution PNG and explicitly declare a continuous target geometry (e.g., `--target_res 100 50`) and watch the model instantly solve the upscaling PDE dynamically.

```bash
# Example: Zero-shot mesh upscaling from 40x20 to 100x50
python eval_model.py \
    --model_name fno_base \
    --input_image ./CNT_40x20/CNT_40x20/125.png \
    --target_res 100 50
```

---

## 🧮 Classical Solvers (Ground Truth Generators)

To build robust training data, the codebase includes traditional TopOpt mathematical solvers integrating purely with `scipy.sparse`. 

*   **Single-Material Optimization**: Python port of Sigmund's classic 99-line SIMP code (`src/top99_2d.py`).
*   **Multi-Material Optimization**: Implementation of the "Alternating Active-Phase" solver for distinct phases + void space (`src/multitop_2d.py`).

---

## 🎯 Future Work & Outlook

The transition to Fourier Neural Operators explicitly solves the resolution constraints of classical ML in the TopOpt problem setups. 

The next frontier for this codebase will explore combining **Latent Diffusion Models** heavily guided by pre-trained FNO / DeepONet physics discriminators. By utilizing diffusion to handle the generative "creativity" and the continuous Operator Learning network acting as a rigorous physics-informed loss function, we aim to match the exact mechanics of traditional iterative solvers at orders of magnitude higher speed.

***

### References
1. **FNO Theory:** Li et al. (2020), *Fourier Neural Operator for Parametric Partial Differential Equations*.
2. **Classic SIMP:** Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab*. 
3. **Multi-Phase:** Tavakoli, R., & Mohseni, S. M. (2014). *Alternating active-phase algorithm for multimaterial topology optimization problems*.