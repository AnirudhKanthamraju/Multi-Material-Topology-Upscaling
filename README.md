# 2D Multi-Material Topology Optimisation

This repository is dedicated to reproducing the datasets and baseline models described in my thesis work. It provides a Python-based framework for 2D Topology Optimisation (TopOpt), implementing both classical single-material optimization and advanced multi-material optimization algorithms.

---

## 🚀 Capabilities

The codebase provides traditional mathematical solvers for structural engineering problems, aiming to maximize stiffness (minimize compliance) for a given amount of material. These solvers act as the ground truth for our dataset generation.

*   **Single-Material Optimization**: A faithful Python port of Sigmund's classic 99-line SIMP code.
*   **Multi-Material Optimization**: Implementation of the "Alternating Active-Phase" algorithm, allowing for designs using multiple materials (e.g., stiff, medium, soft) alongside void space.
*   **Integrated FEM Solver**: Optimized 2D Finite Element Method (FEM) solver using `scipy.sparse` for rapid equilibrium calculations.
*   **Interactive Visualisation**: Real-time plotting of the optimization process and material distribution.

---

## 📂 Codebase Structure

The repository focuses purely on the 2D solvers to generate and validate training data:

```text
├── src/                        # Core Solvers & Utilities
│   ├── top99_2d.py             # 2D Single-Material (SIMP) Solver
│   ├── multitop_2d.py          # 2D Multi-Material (Alternating Phase) Solver
│   ├── FEM_models.py           # Shared FEM & Filter logic (Assembly, Stiffness)
│   └── visualisation.py        # Plotting utilities for 2D structures
├── assets/                     # Images and dataset examples
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (excluding 3D and legacy network files)
└── README.md                   # Project documentation
```

---

## 🛠️ Usage Examples   

### 1. Single-Material Optimization
Use this to find the optimal shape for a single solid material within a defined space.

```bash
# Run with default settings (60x20 cantilever beam)
python src/top99_2d.py

# Run with custom mesh size and volume fraction (40% material)
python src/top99_2d.py --nelx 80 --nely 40 --volfrac 0.4
```

### 2. Multi-Material Optimization
Use this to distribute multiple distinct materials (stiffest to softest) across a structure.

```bash
# Run with default 4-phase optimization (3 materials + void)
python src/multitop_2d.py

# Run with a custom 100x50 mesh and increased iterations
python src/multitop_2d.py --nx 100 --ny 50 --maxiter 300
```

---

## 📚 References

The algorithms implemented in this codebase are based on the following foundational papers:

1. **99-Line Code (Single Material):** 
   Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab*. Structural and Multidisciplinary Optimization, 21(2), 120-127.
2. **Multi-Material Code:**
   Tavakoli, R., & Mohseni, S. M. (2014). *Alternating active-phase algorithm for multimaterial topology optimization problems: a 115-line MATLAB implementation*. Structural and Multidisciplinary Optimization, 49(4), 621-642.
3. **Author's Related Work:**
   Kanthamraju, A., Duriez, E., James, K., & Morlier, J. (2022). *Upscaling optimal topology multimaterials structures using Deep Neural Networks*. CSMA 2022 - 15ème Colloque National en Calcul des Structures. [⟨hal-03693236⟩](https://hal.science/hal-03717882/)

---

## ⚠️ Limitations: Upscaling Topologies using Image Processing



Previous implementations using Generative Adversarial Networks (GANs) tained on image nets struggled to consistently produce physically viable structures. Despite perceptually generating seemingly optimal topologies, the outputs have serious noise leading for currently un-usuble outputs. 

<p align="center">
  <img src="assets/experiment1%20outputs.png" alt="Exp 1 Upscaling Cantilevers on 40x20 grid" width="800"/>
  <br>
  <em>Experiment 1: Upscaling Cantilevers on a 40x20 grid</em>
</p>

<p align="center">
  <img src="assets/experiment2%20outputs.png" alt="Exp 2 Upscaling Cantilevers on 100x50 grid" width="800"/>
  <br>
  <em>Experiment 2: Upscaling Cantilevers on a 100x50 grid</em>
</p>

<p align="center">
  <img src="assets/experiment3%20outputs.png" alt="Exp 3 Upscaling mix of MBB and Cantilevers on 40x20 grid" width="800"/>
  <br>
  <em>Experiment 3: Upscaling mix of MBB and Cantilevers on a 40x20 grid</em>
</p>

---

## 🎯 Conclusion & Future Work

The overarching goal is to dramatically accelerate optimal topology generation while maintaining strict adherence to physical laws. While previous purely image-based GANs fell short, the new approach embraces physics-informed deep learning and generative diffusion models.

**Next Steps & Theoretical Foundation:**

1. **Integrating Fourier Neural Operators (FNO) / DeepONets:**
   Instead of struggling to learn discretized pixel grids, the next architecture will leverage **Operator Learning**. As highlighted in Li et al.'s [Fourier Neural Operator research](https://zongyi-li.github.io/blog/2020/fourier-pde/), FNOs learn the *continuous*, resolution-invariant solution operator for PDE families. By constructing mesh-independent operators, we can achieve zero-shot super-resolution: models trained on a lower resolution can be directly evaluated on a higher resolution, accelerating complex PDE solutions by up to 1000x compared to traditional solvers. We will also explore state-of-the-art developments in foundation models for PDEs (e.g., [arXiv:2512.01421](https://arxiv.org/abs/2512.01421)).

2. **Latent Diffusion Models for Topology Generation:**
   Conventional iterative Topology Optimisation models (like SIMP) start uniformly or from a completely noisy input and slowly "carve out" an optimal design iterative step by iterative step. This maps perfectly onto the mathematical framework of [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion). The future work will use stable diffusion processes guided by DeepONets. Instead of blindly generating images, the diffusion model will iteratively refine pure noise into structurally sound, optimal designs, relying on the pre-trained FNOs to act as a surrogate precision loss function.

This hybrid approach expects to effectively blend the iterative generative creativity of diffusion models with the strict, governing physical precision of traditional FEM solvers. 