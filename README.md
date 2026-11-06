# QuantumDots
# Quantum Dot Finite Difference Method

A finite difference method (FDM) based quantum dot system solver using the effective mass approximation. It computes ground-state energies and electron-hole distributions of many-body quantum dot systems via second quantization.

## Features

- **Multiple geometries**: Sphere, cylinder, rectangle, ellipsoid
- **Core/shell structures**: Supports multi-layer core-shell quantum dot modeling
- **PML boundary conditions**: Perfectly Matched Layers for open boundary problems
- **High-performance numerics**:
  - Numba JIT-accelerated sparse matrix construction
  - PyAMG algebraic multigrid preconditioner for Poisson equation
  - Sparse matrix diagonalization for ground-state solving
- **Coulomb interactions**: Electron-electron, hole-hole, and electron-hole Coulomb matrix elements
- **Second quantization**: Many-body Hamiltonian construction and ground-state solution
- **Electron-hole distribution analysis**: Cosine and difference similarity between electron and hole density distributions

## File Structure

| File | Description |
|------|-------------|
| `qd_fdm.py` | Core module containing the `QuantumDotSystemFDM` class |
| `example.py` | Main calculation script with parameter setup and workflow |

## Dependencies

- Python 3.x
- NumPy
- SciPy (sparse matrices, eigenvalue solvers, conjugate gradient)
- Numba (JIT acceleration)
- PyAMG (algebraic multigrid preconditioner)
- Matplotlib (optional, for visualization)

Install dependencies:

```bash
pip install numpy scipy numba pyamg matplotlib
```

## Quick Start

### Basic Usage

```python
from qd_cal_single import main

# Quantum dot type
qd_type = 'rectangle'

# Geometry parameters (rectangle: [x half-width, y half-width, z half-width])
geo = [[2, 2, 2]]

# Material parameters (core/shell, two layers)
me_list = [0.5, 0.5]      # Electron effective mass (m0)
mh_list = [0.5, 0.5]      # Hole effective mass (m0)
Ve_list = [0.0, 1e8]      # Electron confinement potential (meV)
Vh_list = [0.0, 1e8]      # Hole confinement potential (meV)
eps_r_list = [10, 10]      # Relative dielectric constant

# Calculation parameters
Norb_max = 2               # Maximum number of orbitals
box_size = [10, 10, 10]    # Computational box size (nm)
ini_grid_step = 0.1        # Initial grid step size (nm)
N_stretch = 20             # PML layers per side
s_max = 50                 # PML maximum stretch factor
p = 2                      # PML polynomial order

result = main(qd_type, geo,
              me_list, mh_list, Ve_list, Vh_list, eps_r_list,
              Norb_max=Norb_max, box_size=box_size, ini_grid_step=ini_grid_step,
              N_stretch=N_stretch, s_max=s_max, p=p)
```

### Run Directly

```bash
python qd_cal_single.py
```

## Parameter Reference

### Geometry Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `qd_type` | str | Quantum dot shape: `'sphere'`, `'cylinder'`, `'rectangle'`, `'ellipsoid'` |
| `geo` | list | Geometry dimensions, one entry per shell layer |

Geometry format per shape:
- **Sphere**: `[[r]]` — r is the radius
- **Cylinder**: `[[r, h]]` — r is the base radius, h is the half-height
- **Rectangle**: `[[lx, ly, lz]]` — half-widths along each axis
- **Ellipsoid**: `[[a, b, c]]` — half-axis lengths

Multi-shell example: `[[r_core], [r_shell1], [r_shell2]]`

### Material Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `me_list` | Electron effective mass (by material index) | m₀ (electron rest mass) |
| `mh_list` | Hole effective mass | m₀ |
| `Ve_list` | Electron potential energy offset | meV |
| `Vh_list` | Hole potential energy offset | meV |
| `eps_r_list` | Relative dielectric constant | dimensionless |

Index 0 corresponds to the quantum dot core; indices 1, 2, ... correspond to successive shell layers.

### PML Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_stretch` | 20 | Number of PML layers per side |
| `s_max` | 180 | Maximum stretch factor (recommended: 20–180) |
| `p` | 2 | Polynomial order (recommended: 2–3) |

### Calculation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Norb_max` | 8 | Maximum number of orbitals (controls basis size) |
| `box_size` | [10, 10, 10] | Computational domain size (nm) |
| `ini_grid_step` | 0.1 | Grid step size (nm) |

## Physical Model

### Single-Particle Hamiltonian

Effective mass approximation:

$$H = -\frac{\hbar^2}{2} \nabla \cdot \left(\frac{1}{m^*(\mathbf{r})} \nabla\right) + V(\mathbf{r})$$

where $m^*(\mathbf{r})$ is the position-dependent effective mass and $V(\mathbf{r})$ is the confinement potential.

### Coulomb Interaction

The Coulomb potential is obtained by solving the Poisson equation:

$$-\nabla \cdot (\varepsilon \nabla \phi) = \rho$$

Coulomb matrix elements:

$$V_{ijkl} = \int \rho_{ij}(\mathbf{r}) \phi_{kl}(\mathbf{r}) \, d^3r \times \frac{e^2}{\varepsilon_0}$$

### Second Quantization

Many-body Hamiltonian:

$$H = \sum_i \epsilon_i c_i^\dagger c_i + \frac{1}{2}\sum_{ijkl} V_{ijkl} c_i^\dagger c_j^\dagger c_l c_k$$

## Output

The program returns a tuple containing:

- **Geometry parameters**: Input geometry dimensions
- **Material parameters**: Input material parameter lists
- **Energy levels**: First 4 electron and hole energy levels (meV)
- **Physical quantities**:
  - 1e1h binding energy (meV)
  - Electron-hole cosine similarity
  - Electron-hole difference similarity

Results are automatically saved as a pickle file (`{qd_type}.pkl`).

## Physical Constants

Built-in constants (typical values for III-V quantum dots):

| Constant | Value | Unit |
|----------|-------|------|
| ħ²/(2m₀) | 38.0998 | meV·nm² |
| e²/ε₀ | 18095.128 | meV·nm |

## Notes

1. **Grid convergence**: The grid step `ini_grid_step` affects accuracy; convergence tests are recommended.
2. **PML parameters**: `s_max` and `N_stretch` must be sufficiently large for proper boundary absorption.
3. **Orbital truncation**: `Norb_max` controls the accuracy–efficiency trade-off.
4. **Material parameters**: Ensure effective masses are positive and the material list length matches the number of shell layers.
5. **Memory usage**: Large grids and high orbital counts significantly increase memory consumption.

## Citation

This repository can be cited using:
```
@article{casimir_QD,
  author = {Tang, Kun and Huang, Peng and Zhang, Yongyou and Zhong, Haizheng},
    title = {Casimir-like Excitonic States from Twin-Structured Type-II Core/Shell Quantum Dots},
    journal = {Nano Letters},
    year = {2026},
    month = {11},
    doi = {10.1021/acs.nanolett.6c04338},
    url = {https://doi.org/10.1021/acs.nanolett.6c04338},
    eprint = {https://pubs.acs.org/nalefd/article-pdf/doi/10.1021/acs.nanolett.6c04338/69066454/acs.nanolett.6c04338.pdf},
}
```
