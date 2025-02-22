# Steering Vector Coefficient Selection

This repository contains research and implementation for automatically determining optimal coefficients when using steering vectors for language model control. The work focuses on finding reliable metrics and methods to detect when steering transitions from effective to incoherent.

## Findings

My research demonstrates that average log probabilities provide the most reliable indicator for steering effectiveness and model coherence. The method works by:
1. Running a model across a range of coefficients
2. Detecting the transition point where generation becomes incoherent
3. Automatically selecting optimal coefficients below this threshold

For detailed methodology and results, see the full writeup in `/report/steering-coefficient.md`.

## Notebooks

- **Interactive Demo**: A demo that runs the coefficient selection algorithm and lets you generate with a steered base or instruct model: `experiments/notebooks/interactive_demo.ipynb`.
  ![image](https://github.com/user-attachments/assets/a9f46aaa-b1c1-48ae-becd-350c6b685d23)


- **Results & Analysis**: Comprehensive evaluation of different metrics and methods for coefficient selection, including comparisons with existing approaches: `experiments/notebooks/results.ipynb`.

- **Autograding**: Implementation of LLM-based grading for steering vector effectiveness: `experiments/notebooks/autograding_steering.ipynb`. Note this is not included in the report, as it's stil a work in progress.

## Requirements

- Python 3.10+
- Hugging Face API key with access to Llama 3 models
- For autograding: Anthropic API key
- GPU requirements (not optimized, sorry):
  - Interactive demo: 93GB+ VRAM (H100 NVL) or 2x A100 80GB
  - Results & autograding notebooks: 2x A100 80GB

## Credits

Thank you to @vooooogel for the excellent repeng library, which this is based on.

Thank you to Plastic Labs for sponsoring this research.
