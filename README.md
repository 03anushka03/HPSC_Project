# HPSC_Project
# 2D Ising Model Monte Carlo Simulation with OpenMP Parallelization

This project implements the 2D Ising Model using the Metropolis Monte Carlo algorithm in C++ and parallelizes the computational workflow using OpenMP for High Performance Scientific Computing (HPSC).

The Ising Model is a fundamental model in statistical mechanics used to study ferromagnetism, critical phenomena, and second-order phase transitions. Each lattice site contains a spin that can take values of +1 or -1, and nearest-neighbor interactions determine the total system energy.

The simulation investigates how magnetization and energy vary with temperature and captures the phase transition near the critical temperature:

Tc ≈ 2.269

---

## Project Objectives

* Develop a serial implementation of the 2D Ising Model
* Verify physical correctness of the simulation
* Profile runtime using C++ chrono
* Study compiler optimization effects using `-O0`, `-O2`, and `-O3`
* Parallelize the simulation using OpenMP
* Perform performance analysis using runtime, speedup, and efficiency plots

---

## Physics Background

The Hamiltonian of the 2D Ising Model is:

H = -J Σ sᵢsⱼ

where:

* J = interaction strength
* sᵢ = spin at lattice site i (+1 or -1)
* nearest-neighbor interactions are considered

The Metropolis algorithm is used for Monte Carlo updates based on the Boltzmann acceptance probability.

---

## Repository Structure

### `Ising_serial.cpp`

Serial implementation of the 2D Ising Model using the Metropolis Monte Carlo method.

Includes:

* lattice initialization
* energy calculation
* magnetization calculation
* Metropolis updates
* temperature sweep
* output generation

---

### `Ising_parallel.cpp`

OpenMP parallel implementation of the Ising simulation.

Parallelization strategy:

* temperature-level parallelism using OpenMP

Used for:

* runtime scaling
* speedup analysis
* efficiency analysis

---

### `plot_results.py`

Python script for generating plots from simulation output.

Generates:

* Magnetization vs Temperature
* Energy vs Temperature
* Runtime vs Threads
* Speedup vs Threads
* Efficiency vs Threads
* Optimization vs Runtime

---

### `Plots/`

Contains all generated publication-quality figures used in the final report.

---

### `Makefile`

Build instructions for both serial and parallel versions.

---

### `Report.tex`

IEEE-format final white paper for project submission.

Includes:

* formulation
* verification
* profiling
* OpenMP implementation
* performance analysis
* conclusions

---

## Compilation

### Serial Code

```bash
g++ -O3 Ising_serial.cpp -o ising_serial
./ising_serial
```

### Parallel Code

```bash
g++ -fopenmp -O3 Ising_parallel.cpp -o ising_parallel
export OMP_NUM_THREADS=4
./ising_parallel
```

---

## Results

The simulation successfully demonstrates:

* ordered ferromagnetic phase at low temperature
* disordered paramagnetic phase at high temperature
* phase transition near Tc ≈ 2.269

OpenMP parallelization improves runtime significantly and allows performance scaling analysis using multiple threads.

---

## Tools Used

* C++
* OpenMP
* Python (Matplotlib, Pandas)
* IEEE Conference LaTeX Template

---

## Course Information

This project was developed as part of the High Performance Scientific Computing (HPSC) course and focuses on both scientific correctness and computational performance optimization.
