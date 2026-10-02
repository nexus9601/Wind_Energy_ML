# Wind Energy ML

> **Temporary project overview:** This README is a working draft and will be updated as the analysis and modeling workflow take shape.

## Project Overview

This project explores wind resource data for India as a starting point for wind-energy analysis and machine-learning experiments. The current work is focused on preparing the source data and performing exploratory data analysis; predictive modeling has not yet been documented.

## Data

The raw dataset is NIWE's 150 m India Wind Resource Map data. It describes wind-resource characteristics at 500 m grid resolution, including wind speed, Weibull parameters, wind power density, temperature, air density, and wind direction. See [`data/raw/metadata_150m.txt`](data/raw/metadata_150m.txt) for the source-provided description.

- Dataset file: `data/raw/150m_Map_Data_A_to_G.csv`
- Format: CSV
- Reported size: approximately 836.64 MB

The source dataset is large. Keep raw data in `data/raw/` and place any derived or cleaned outputs in `data/processed/`.

## Current Work

[`notebooks/02_data_cleaning_and_eda.ipynb`](notebooks/02_data_cleaning_and_eda.ipynb) contains the current data-cleaning and exploratory-analysis workflow. It reads the CSV in chunks and samples rows for EDA to avoid loading the full dataset into memory. The notebook currently includes Google Colab setup for accessing/downloading the raw data.

## Project Status

- Raw dataset and metadata are present in the project workspace.
- Chunked sampling and initial data inspection are in progress.
- Modeling goals, evaluation criteria, and final outputs are to be determined.