# Crossover from Discrete to Continuous Quantum Geometry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23195881.svg)](https://doi.org/10.5281/zenodo.23195881)

Code, data, figures and manuscript for

> **R. Chen**, *Crossover from Discrete to Continuous Quantum Geometry: Running Spectral Dimensions,
> Noncommutative Spectral Triples, and Critical Phase Transitions* (2026).
> DOI: [10.5281/zenodo.23195881](https://doi.org/10.5281/zenodo.23195881)

## Summary

- **Spectral dimension as equipartition.** d_s(τ) = 2τ⟨λ⟩_τ exactly. Any discretisation with a bounded spectrum gives
  d_s → 0 as τ → 0 (a cutoff artefact), and a finite graph gives d_s → 0 as τ → ∞.
- **Exact running dimensions.** Hypercubic lattice: d_s = 2d x[1 − I₁(x)/I₀(x)], x = 2τ/a², which overshoots d
  (maximum 1.2178 d) and approaches d from above. Higher-derivative dispersion E = k²(1 + ℓ²k²) in d = 4: closed form
  P ∝ τ⁻¹[1 − √π s e^{s²} erfc(s)], s = √τ/2ℓ, running from d_s = 2 to 4 (general d via parabolic-cylinder functions).
- **Dimensional splitting.** d_s = 2d_H/d_w; the Sierpinski gasket gives d_s = 2 ln3/ln5 with log-periodic oscillations
  of period ln 5.
- **Spectral triples.** The lattice Hodge–Dirac operator d + d* on T^d converges in norm-resolvent sense with error
  O(a); the naive lattice Dirac operator does not (fermion doubling). For graph Dirac operators the Connes distance
  is bounded between a·d_hop/√deg and a·d_hop; on ℤ^d the hop metric tends to ℓ¹, while lower bounds on the Connes
  distance converge to the Euclidean distance.

This is the fourth paper of a series:
[1](https://doi.org/10.5281/zenodo.23189575) holographic entanglement of fractal regions
([code](https://github.com/Ruqing1963/holographic-fractal-entanglement)),
[2](https://doi.org/10.5281/zenodo.23193512) multifractal large deviations in quantum entanglement
([code](https://github.com/Ruqing1963/multifractal-quantum-entanglement)),
[3](https://doi.org/10.5281/zenodo.23194419) uncomputable fractal dimensions and the Chaitin barrier
([code](https://github.com/Ruqing1963/chaitin-fractal-complexity)).

## Repository structure

| Path | Contents |
|---|---|
| `paper/` | LaTeX source and compiled PDF of the manuscript |
| `code/quantum_geometry_crossover.py` | Single script producing every number and figure in the paper |
| `figures/` | Figures as vector PDF (used by the paper) and PNG |
| `data/` | Numerical data as CSV (metadata in `#` header lines) |
| `results/` | Console output of the script (the numbers quoted in the paper) |

## Reproducing the results

Requirements: Python ≥ 3.10 and the packages in `requirements.txt`
(tested with Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.8.4). Runtime is about 20 seconds.

```bash
pip install -r requirements.txt
python code/quantum_geometry_crossover.py      # add --show to display the figures
```

Run from the repository root; figures go to `figures/` and data to `data/` (override with `QGC_FIG_DIR`,
`QGC_DATA_DIR`).

| Data file | Content | Paper |
|---|---|---|
| `square_lattice_heat_kernel.csv` | P(τ), d_s (finite difference, exact, infinite-lattice formula) for the 64×64 lattice | §2.2, Fig. 1 |
| `sierpinski_heat_kernel.csv` | P(τ), d_s for the level-7 Sierpinski gasket | §3.2, Figs. 1–2 |
| `modified_dispersion_d4.csv` | d_s for E = k²(1+ℓ²k²) and E = k² with cutoff, and the closed form | §2.3, Fig. 1 |
| `dirac_resolvent_convergence.csv` | resolvent errors of Hodge–Dirac and naive lattice Dirac operators | §4.1, Fig. 2 |
| `connes_distance_bounds.csv` | Connes-distance lower bounds relative to Euclidean and ℓ¹ distances | §4.2, Fig. 2 |

To rebuild the paper (pdfLaTeX, two passes):

```bash
cd paper
pdflatex Chen_2026_Quantum_Geometry_Crossover.tex
pdflatex Chen_2026_Quantum_Geometry_Crossover.tex
```

## Citation

```bibtex
@misc{Chen2026QuantumGeometryCrossover,
  author = {Chen, Ruqing},
  title  = {Crossover from Discrete to Continuous Quantum Geometry: Running Spectral Dimensions,
            Noncommutative Spectral Triples, and Critical Phase Transitions},
  year   = {2026},
  doi    = {10.5281/zenodo.23195881},
  url    = {https://doi.org/10.5281/zenodo.23195881}
}
```

## License

- **Code** (`code/`): [MIT License](LICENSE)
- **Manuscript, figures, data and results** (`paper/`, `figures/`, `data/`, `results/`):
  [CC BY 4.0](LICENSE-CC-BY-4.0.md)

## Contact

Ruqing Chen — GUT Geoservice Inc., Montreal — ruqing@hotmail.com
