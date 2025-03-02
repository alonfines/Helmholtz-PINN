# **Physics-Informed Neural Network (PINN) for 2D Helmholtz Equation**

<p align="center">
  <img src="https://github.com/user-attachments/assets/b14e399c-1ede-4fa0-ab1e-3d826e2e8ba3" width="36%" alt="Curriculum Training Visualization">
  <img src="https://github.com/user-attachments/assets/b9f9263f-cecf-4ce9-bc91-91d82972eb79" width="30%" alt="Seq2Seq Training Visualization">
  <img src="https://github.com/user-attachments/assets/6ff6f317-d512-4e9f-b05b-6183bfee6d4d" width="30%" alt="Comparison of PINN and FEM Solutions">
</p>

<p align="center">
  <i> FEM Solver | PINN Curriculum | PINN Seq2seq</i>
</p>

## **Overview**
This project implements a **Physics-Informed Neural Network (PINN)** to solve the **2D Helmholtz equation** using curriculum and seq2seq methods. 
These methods can be further detailed in [1].
We employ **Finite Element Method (FEM) simulations** for generating initial condition data and comparison results and utilize PINN to learn the wave field solution.

---

## **Problem Definition**
The **2D homogeneous Helmholtz equation** is given by:

$$
\Delta U + k^2 U = 0
$$

where:
- $$\ U(x, z) \$$ represents the wave field.
- $$\ k^2 \$$ is the wavenumber squared.
- Here we focus on **Absorbing Boundary Conditions (ABC)** which are applied to model realistic wave propagation scenarios.

---

## **Workflow**

### **Step 1: Geometry Setup and FEM Solver (MATLAB)**
- The problem domain and wave propagation setup are defined using MATLAB-based **Finite Element Method (FEM)** scripts.
- The primary solver is based on:
  
  **David Gasperini (2025),** *"FEM Solver for 2D Helmholtz Equation"*, available at:  
  [MATLAB Repository](https://www.mathworks.com/matlabcentral/fileexchange/91695-fem-solver-for-2d-helmholtz-equation)

- **Modifications:**
  - The script `Diffraction2.m` was implemented to model wave propagation through a **slit**.
  - The computed wave field data (real and imaginary components) is stored in `grid_data.mat`.

---

### **Step 2: Data Processing (Python)**
- The **MATLAB output** (`grid_data.mat`) is processed in Python using `Notebook.ipynb`.
- The dataset is converted into CSV format for compatibility with the PINN model.

#### **Extracted Features:**
- Either **real** or **imaginary** values of \( U(x, z) \) are extracted and saved as a CSV file.
- Boundary conditions at \( z=0 \) are stored separately in **`bc1.csv`**.

These files must be placed in the **`boundary_conditions`** directory within either the `curriculum` or `seq2seq` folders.

---

## **PINN Implementation**

The PINN is implemented using **PyTorch Lightning**.
There is a curriculum dir applying the curriculum method and seq2seq dir applying the seq2seq method.

### **Key Components**
1. **Configuration Files** (`config_curriculum.yaml`, `config_seq2seq.yaml`):
   - Define **hyperparameters** such as:
     - Learning rate
     - Batch size
     - Number of epochs
     - Computational domain parameters (e.g., \( L_x, z \))
     - Training error thresholds

2. **Dataset Handling** (`dataset_curriculum.py`, `dataset_seq2seq.py`):
   - Loads training points from a specified computational domain.

3. **Model Definition** (`model_curriculum.py`, `model_seq2seq.py`):
   - Implements a **fully connected neural network (FCNN)** with **Tanh** activation functions.
   - Computes **Helmholtz residual loss** and **boundary condition loss**.

4. **Training Scripts** (`train_curriculum.py`, `train_seq2seq.py`):
   - Loads the dataset and trains the PINN using **Adam optimizer** with adaptive weight balancing.
   - Uses **WandB (Weights & Biases)** for logging.

5. **Testing & Evaluation** (`test_curriculum.py`, `test_seq2seq.py`):
   - Loads trained models for evaluation.
   - Compares **PINN solutions** with **FEM reference solutions**.
   - Computes **L2 error** and generates visualization plots.

6. **Boundary Condition Generation** (`bc_creator.py`):
   - Generates **boundary conditions** dynamically for sequential training.

---

## **Results & Visualization**
- The trained model predictions are **compared with FEM solutions**.
- Checkpoint files are saved in the **`checkpoints`** directory.
- Visualization plots are saved in the **`plots`** directory.
- **L2 error** between PINN and FEM solutions is computed.

---

## **References**
[1] Krishnapriyan, A.S., Gholami, A., Zhe, S., Kirby, R.M., & Mahoney, M.W. (2021). Characterizing possible failure modes in physics-informed neural networks.

[2] Gasperini, D. (2025). *FEM Solver for 2D Helmholtz Equation*.

---
