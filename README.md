# BBH Merger Detection — Data Preprocessing Pipeline
Research-oriented gravitational-wave preprocessing framework using GWOSC and GWpy.

### Overview

This repository contains a modular preprocessing pipeline for gravitational-wave strain data from confirmed Binary Black Hole (BBH) merger events observed during the LIGO–Virgo O3 observing run.

The goal of this project is to build a clean, reproducible, and research-oriented preprocessing framework for gravitational-wave signal analysis using publicly available data from the Gravitational Wave Open Science Center (GWOSC).

The pipeline is being developed step-by-step from first principles, with emphasis on:

- transparent preprocessing decisions
- reproducibility
- numerical stability
- modular design
- compatibility with downstream machine learning workflows

### Scientific Motivation

Gravitational-wave detectors such as LIGO measure extremely small spacetime strain signals produced by astrophysical events like Binary Black Hole mergers.

Raw detector strain data contains:

- instrumental noise
- low-frequency seismic contamination
- high-frequency noise
- detector artifacts
- varying signal amplitudes

Before any detection or machine learning model can be applied, the data must be carefully preprocessed.

This repository focuses specifically on building that preprocessing stage.


### Data Source

This project uses publicly available gravitational-wave strain data from:

- GWOSC (Gravitational Wave Open Science Center)

Event metadata and detector strain are accessed through:

- gwosc
- gwpy






