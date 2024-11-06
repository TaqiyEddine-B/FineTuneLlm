# Fine-Tuning Llama-3.1-8B-Instruct

This repository contains code for fine-tuning [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on the [Bitext - Customer Service Tagged Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

For a complete walkthrough and technical details, check out the blog post.

## Quick Start
The Makefile contains the main commands to run the fine-tuning process :

1. Create a virtual environment:
```bash
make setup
```

2. Install dependencies:
```bash
make install
```

3. Launch the fine-tuning process:
```bash
make train
```
