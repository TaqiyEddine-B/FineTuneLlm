import datetime
import os

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from zoneinfo import ZoneInfo

from config import FinetuneConfig
from utils import setup_mlflow_tracking


class LlmTrainerUnsloth:
    def __init__(self, model_name: str, dataset_name: str):
        """Initialize the LlmTrainer with Unsloth class."""
        self.config = FinetuneConfig()

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.max_seq_length = 2048

        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        self.model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.r,
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=42,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

    def load_dataset(self, dataset_name: str, percentage: float = 1.0):
        """Load and preprocess the dataset."""

        dataset = load_dataset(dataset_name, split="train")
        print(f"Loading original dataset: {len(dataset)} samples")

        # Calculate the number of samples to select based on the percentage
        sample_size = int(len(dataset) * percentage)
        self.dataset = dataset.select(range(sample_size))
        print(f"Loaded dataset: {len(self.dataset)} samples")

    def configure_trainer(self, experiment_name):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            dataset_text_field="response",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                max_steps=self.config.max_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                optim=self.config.optim,
                seed=self.config.seed,
                save_strategy=self.config.save_strategy,
                logging_steps=self.config.logging_steps,
                output_dir=f"output/{experiment_name}",
                report_to=["mlflow"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
            ),
        )

    def train_model(self, experiment_name: str = "llama_experiment"):
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
        model_short_name = model_name.split("/")[-1]
        timestamp = datetime.datetime.now(tz=ZoneInfo("UTC")).strftime("%Y%m%d_%H%M")

        experiment_name = f"{model_short_name}_finetune_{timestamp}"

        self.load_dataset(dataset_name=self.dataset_name)

        self.configure_trainer(experiment_name)

        self.train_model(experiment_name=experiment_name)

        self.save_model(experiment_name=experiment_name)


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

    LlmTrainerUnsloth(model_name=model_name, dataset_name=dataset_name).train_pipeline()
