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

3. Authenticate with Hugging Face
You need to authenticate your environment with Hugging Face to access the model. You can do this by logging in using the Hugging Face CLI:
```bash
huggingface-cli login
```
This command will prompt you to enter your Hugging Face credentials. Make sure you use the account that has access to the gated model.


4. Launch the fine-tuning process:
```bash
make train
```
