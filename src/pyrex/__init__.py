"""
Pyrex
=====

Pyrex is a lightweight toolkit for extending quasi-circular gravitational
waveforms with eccentricity. It implements the complete eccentricity
reconstruction pipeline needed to produce consistent eccentric waveforms from
existing circular models.

Features
--------
- Load pre-fitted eccentric-modulation parameters from disk.
- Interpolate modulation coefficients across mass ratio, eccentricity, and
  PN-like x-values.
- Reconstruct early-time eccentric amplitude and phase.
- Stitch reconstructed eccentric segments to a circular baseline waveform.
- Apply smoothing and continuity corrections.
- Provide a simple interface for generating eccentric waveforms for analysis,
  simulation, or machine-learning workflows.

The package is model-agnostic: any quasi-circular waveform class that exposes
phase, amplitude, and frequency arrays can be extended with eccentricity using
the Pyrex utilities.
"""
