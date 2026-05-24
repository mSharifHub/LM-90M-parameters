# Building a Large Language Model (LLM)

This project focuses on the step-by-step process of building a Large Language Model from scratch.

## Current Progress

### 1. Data Preparation
- **Dataset:** Using `HuggingFaceFW/fineweb-edu` (sample-10BT).
- **Loading:** Data is streamed and a subset (10,000 rows) is saved locally in Hugging Face format.
- **Tokenization:**
    - Implementing a custom regex-based tokenizer.
    - Pattern used: `r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""`
    - Parallel processing of rows for efficient tokenization.

## Project Structure
- `preparing_text.ipynb`: Jupyter notebook containing the data loading and initial tokenization logic.
- `local_fineweb_data/`: Local storage for the processed dataset.

## Setup
To run the notebook, ensure you have the following dependencies installed:
```bash
pip install datasets multiprocess
```
