# Cloud Workload Forecasting Challenge

This repository contains the solution for the Atomity AI/ML Engineering Intern forecasting challenge. The goal is to forecast near-term aggregate cloud demand (`active_vm_count` and `aggregate_cpu_utilization`) based on VM telemetry data.

## Environment Setup

Python 3.11+ is required.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install lightgbm matplotlib scikit-learn