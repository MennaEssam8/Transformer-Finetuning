# Transformer Fine-Tuning: Multi-Agent Systems Documentation

A from-scratch implementation of a Transformer block, paired with domain-specific fine-tuning of a small pre-trained LLM (`distilgpt2`) on multi-agent LLM system architecture documentation.

This project demonstrates two things independently: (1) understanding of the core Transformer mechanism, implemented without relying on a high-level library, and (2) the practical fine-tuning workflow used to specialize a pre-trained LLM for a narrow domain.

## Overview

- **Part 1 — Transformer internals:** a `BasicTransformerBlock` built with raw PyTorch (multi-head self-attention + position-wise feed-forward network, pre-norm residual connections, causal masking support), verified with a forward/backward-pass self-test.
- **Part 2 — Domain fine-tuning:** `distilgpt2` fine-tuned on a custom technical corpus covering multi-agent LLM architecture patterns — agent routing, shared state management, tool-calling, human-in-the-loop gates, retrieval-augmented generation, observability, and deployment.
- **Part 3 — Evaluation:** base vs. fine-tuned model outputs compared on an identical prompt, with a written report on the qualitative differences.

## Project Structure

```
specialized-content-generator/
├── fine_tuning_data/
│   ├── custom_dataset.txt          # domain corpus (multi-agent LLM architecture docs)
│   └── generation_samples.json     # raw base vs. fine-tuned generation outputs
├── fine_tuned_model_checkpoint/
│   ├── model_weights.pt            # fine-tuned model state dict
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── src/
│   ├── basic_transformer_block.py  # from-scratch Transformer block + self-test
│   └── fine_tuning_script.py       # end-to-end fine-tuning pipeline
├── comparison_report.md            # base vs. fine-tuned qualitative analysis
└── README.md
```

## Tech Stack

- Python 3.12
- PyTorch
- Hugging Face `transformers` (`distilgpt2`)
- Hugging Face `tokenizers` (offline BPE fallback — see note below)

## Setup

```bash
git clone https://github.com/MennaEssam8/transformer-finetuning-multiagent-docs.git
cd transformer-finetuning-multiagent-docs
pip install torch transformers tokenizers
```

## Usage

**Run the Transformer block self-test:**

```bash
python src/basic_transformer_block.py
```

Verifies shape-preservation, causal masking, and gradient flow through a randomly initialized block.

**Run the fine-tuning pipeline:**

```bash
python src/fine_tuning_script.py
```

This will:
1. Load pre-trained `distilgpt2` from the Hugging Face Hub.
2. Generate 5 baseline completions for a fixed evaluation prompt.
3. Fine-tune on `fine_tuning_data/custom_dataset.txt` for 5 epochs.
4. Save the fine-tuned checkpoint to `fine_tuned_model_checkpoint/`.
5. Generate 5 post-fine-tuning completions on the same prompt.
6. Save all samples to `fine_tuning_data/generation_samples.json`.

Hyperparameters (learning rate, batch size, block size, weight decay) are documented inline in `fine_tuning_script.py` with rationale for each choice.

## Results

See [`comparison_report.md`](./comparison_report.md) for the full qualitative comparison of base vs. fine-tuned outputs, including training loss curves and a discussion of style, coherence, and domain adherence.

## A Note on Reproducibility

`fine_tuning_script.py` includes an automatic offline fallback: if the Hugging Face Hub is unreachable (e.g. a network-restricted CI/sandbox environment), it substitutes a small randomly-initialized model of the same architecture with a locally-trained tokenizer, so the full pipeline can still be exercised end-to-end without internet access. On any machine with normal internet access, it transparently uses the real pre-trained `distilgpt2` — no configuration needed. `comparison_report.md` documents exactly which mode produced its results.

## Author

**Mennatullah Essam** — AI Engineer
[GitHub](https://github.com/MennaEssam8) · [LinkedIn](https://linkedin.com/in/mennaessam28)
