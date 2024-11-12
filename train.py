import datetime
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments
from trl import SFTTrainer
from zoneinfo import ZoneInfo

from config import FinetuneConfig
from utils import setup_mlflow_tracking


class LlmTrainer:
    def __init__(self, model_name: str, dataset_name: str):
        """Initialize the LlmTrainer class."""

        self.config = FinetuneConfig()

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.trainer = None

    def load_dataset(self, dataset_name: str, percentage: float = 1.0):
        """Load and preprocess the dataset."""

        dataset = load_dataset(dataset_name, split="train")
        print(f"Loading original dataset: {len(dataset)} samples")

        # Calculate the number of samples to select based on the percentage
        sample_size = int(len(dataset) * percentage)
        self.dataset = dataset.select(range(sample_size))
        print(f"Loaded dataset: {len(self.dataset)} samples")

    def configure_trainer(self, experiment_name: str):
        """Configure the trainer with the specified configurations."""

        self.training_args = TrainingArguments(
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            seed=self.config.seed,
            save_strategy=self.config.save_strategy,
            output_dir=f"output/{experiment_name}",
            logging_steps=self.config.logging_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=["mlflow"],
        )

        # LoRA configuration for model adaptation
        self.lora_config = LoraConfig(r=self.config.r)

    def train_model(self, experiment_name: str = "llama_experiment"):
        # Fine-tune model within an mlflow experiment

        # Initialize the trainer with configurations
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            peft_config=self.lora_config,
            train_dataset=self.dataset,
            dataset_text_field="response",
        )
        print("Starting model fine-tuning...")
        with setup_mlflow_tracking(self.model_name) as _:
            self.trainer.train()
            print("Training completed.")

    def save_model(self, experiment_name: str):
        # Save the trained model to the specified path

        output_dir = "output/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, experiment_name)

        model_to_save = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        model_to_save.save_pretrained(save_path)

        print(f"Model saved to {save_path}")

    def train_pipeline(self):
        """The main training pipeline."""
        model_short_name = model_name.split("/")[-1]
        timestamp = datetime.datetime.now(tz=ZoneInfo("UTC")).strftime("%Y%m%d_%H%M")
        experiment_name = f"{model_short_name}_finetune_{timestamp}"

        self.load_dataset(dataset_name=self.dataset_name)

        self.configure_trainer(experiment_name=experiment_name)

        self.train_model(experiment_name=experiment_name)

        self.save_model(experiment_name=experiment_name)


if __name__ == "__main__":
    """Main entry point for the script."""
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

    LlmTrainer(model_name=model_name, dataset_name=dataset_name).train_pipeline()
