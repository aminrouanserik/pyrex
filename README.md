# Pyrex

[![DOI](https://zenodo.org/badge/246883158.svg)](https://zenodo.org/badge/latestdoi/246883158)

## Description
Python package for transforming circular gravitational waveforms to low-eccentric waveforms from numerical simulations.

Original implementation by **Yoshinta Setyawati (2021)**.  
Modified and maintained by **Amin Rouan Serik (2025)** to support modern Python packaging and enhancements.

See [Phys. Rev. D 103, 124011](https://doi.org/10.1103/PhysRevD.103.124011) or [arXiv:2101.11033](https://arxiv.org/abs/2101.11033) for details of the model and citation.

---

## Installation

Using `pip` with editable install from your repository (compatible with `pyproject.toml`):

```bash
git clone git@github.com:aminrouanserik/pyrex.git
cd pyrex
pip install -e .
```

## To-Do
- [x] Make Cookware a set of functions, not a class. Test every function individually (including 0 eccentricity test cases)
- [ ] Rewrite Glassware using qcextender (and test)
- [ ] Make sure Glassware uses metadata for q and chi
- [ ] Include back up for q, either take closest or generate one from model
- [ ] Include gweccentricity

## Ideas
- [ ] Boundary conditions for final phase and amp

## Notes
- For every mass ratio there should be a simulation with 0 eccentricity in the training set