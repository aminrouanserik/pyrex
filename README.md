# Pyrex

[![DOI](https://zenodo.org/badge/246883158.svg)](https://zenodo.org/badge/latestdoi/246883158)

## Description
Pyrex provides a lightweight post-processing pipeline that adds eccentricity corrections to quasi-circular gravitational waveforms using interpolation-based fits.
It wraps around standard `lalsimulation` approximants (e.g., IMRPhenomD, TaylorF2) and outputs a modified `qcextender` waveform object.

### Features
- Add eccentricity corrections to circular waveform models
- Drop-in replacement for generating eccentric waveforms
- Smooth hybridization between reconstructed (eccentric) and model (late-time) regions
- Fast interpolation of fit parameters across (q, e, x)
- Works with standard waveform parameters (masses, distance, delta_t, f_lower, etc.)

For the original model and methodology, see [Phys. Rev. D 103, 124011](https://doi.org/10.1103/PhysRevD.103.124011) or [arXiv:2101.11033](https://arxiv.org/abs/2101.11033).

---

## Installation

Using `uv` (recommended):

```bash
git clone git@github.com:aminrouanserik/pyrex.git
cd pyrex
uv sync
```

Using `pip`:

```bash
git clone git@github.com:aminrouanserik/pyrex.git
cd pyrex
pip install -e .
```

---

## Usage Example

```python
from pyrex.eccentric_extension import generate_eccentric_waveform

wf = generate_eccentric_waveform(
    approximant="IMRPhenomD",
    mode=[(2,2)],
    mass_1=30,
    mass_2=20,
    eccentricity=0.15,
    delta_t=1/4096,
    f_lower=20,
)
```
This returns the modified eccentric waveform as a `qcextender` waveform object.

---

## License

MIT License

Original implementation by **Yoshinta Setyawati (2021)**.  
Modernized and refactored by **Amin Rouan Serik (2025)**.

---

## Notes
- For every mass ratio there should be a simulation with 0 eccentricity in the training set
- Pyrex does not generate eccentric waveforms from first principles; it applies interpolation-based corrections on top of circular models.