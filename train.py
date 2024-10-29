import datetime
import os

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, TrainingArguments
from trl import SFTTrainer
from zoneinfo import ZoneInfo


class LlmTrainer:
    def __init__(self, model_name: str, bit_size: int = 8):
        # Configure quantization settings for efficient loading and training

        quant_type = "nf4" if bit_size == 4 else "int8"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=(bit_size == 4),
            load_in_8bit=(bit_size == 8),
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load model with specified quantization settings
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            # quantization_config=self.bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.trainer = None

    def load_dataset(self, dataset_name: str, sample_size: int = 5374):
        # Load the specified dataset and reduce to the desired sample size

        # 5374 is 20% of the dataset
        dataset = load_dataset(dataset_name, split="train")
        print(f"Loading original dataset: {len(dataset)} samples")

        self.dataset_reduced = dataset.select(range(sample_size))
        print(f"Loaded dataset: {len(self.dataset_reduced)} samples")

    def configure_trainer(
        self, output_dir: str = ".results/", logging_dir: str = "./logs", epochs: int = 3, batch_size: int = 8
    ):
        # Set training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_dir=logging_dir,
            logging_steps=10,
            gradient_checkpointing=True,
            report_to=None,
        )

        # LoRA configuration for model adaptation
        lora_config = LoraConfig(r=8, bias="none", task_type="CAUSAL_LM")

        # Initialize the trainer with configurations
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            peft_config=lora_config,
            train_dataset=self.dataset_reduced,
            dataset_text_field="response",
        )

    def train_model(self, experiment_name: str = "llama_experiment"):
        # Fine-tune model within an mlflow experiment
        print("Starting model fine-tuning...")

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as _:
            self.trainer.train()
        print("Training completed.")

    def save_model(self, save_path: str = "output/"):
        # Save the trained model to the specified path

        model_to_save = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        model_to_save.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


# main function
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

    model_short_name = model_name.split("/")[-1]
    timestamp = datetime.datetime.now(tz=ZoneInfo("UTC")).strftime("%H_%M_%d%m%Y")

    experiment_name = f"{model_short_name}_finetune_{timestamp}"

    trainer = LlmTrainer(model_name=model_name)

    trainer.load_dataset(dataset_name=dataset_name)

    trainer.configure_trainer()

    trainer.train_model(experiment_name=experiment_name)

    output_dir = "output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, experiment_name)

    trainer.save_model(save_path)
