# AES CoT NTL

Automated essay scoring experiment code using a pipeline split across training, inference, and evaluation modules.

## Layout

- `main_pipeline.py`: orchestrates training, inference, and evaluation.
- `modules/`: implementation modules for each pipeline stage.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Check dataset paths, model settings, and output directories first, then run:

```bash
python main_pipeline.py
```
