"""Module for base and finetuned model inference classes"""

import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import InferenceConfig


class FinetunedModelInference:
    """Class for finetuned model inference"""

    def __init__(self, base_model_name: str, adapter_path: str):
        self.config = InferenceConfig()

        print(f"Loading base model {base_model_name}")

        # Load the tokenizer for the LLaMA model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

        # Load the base model with specified parameters
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=self.config.torch_dtype,
            device_map="cuda:0",
        )

        # Load and apply the LoRA adapter
        print(f"Loading adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=self.config.torch_dtype,
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.config.torch_dtype,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            batch_size=16,
            max_new_tokens=self.config.max_new_tokens,
        )

    def generate(self, prompt: str):
        """Generate text based on the provided prompt."""
        torch.cuda.empty_cache()  # Clear GPU cache before generation

        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>
        You are Question/answer assistant.
        Your answer must begin with capital letter and end with full stop.
        <|end_header_id|>{prompt}<|eot_id|>
        """

        result = self.pipeline(
            formatted_prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True,  # Clean output
            return_full_text=False,  # Only return new generated text
        )
        return result[0]["generated_text"].strip()


class BaseInference:
    """Class for base model inference"""

    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B"):
        """Initialize the class with model ID"""

        self.config = InferenceConfig()

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with specific GPU configurations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.config.torch_dtype,
            device_map="cuda:0",
        )

        # Create pipeline with GPU specifications
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.config.torch_dtype,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.config.max_new_tokens,
        )

    def generate(self, prompt: str):
        """Generate text based on the provided prompt"""

        torch.cuda.empty_cache()  # Clear GPU cache before generation
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>
        You are Question/answer assistant.
        Your answer must begin with capital letter and end with full stop.
        <|end_header_id|>{prompt}<|eot_id|>
        """

        result = self.pipeline(
            formatted_prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True,  # Clean output
            return_full_text=False,  # Only return new generated text
        )
        return result[0]["generated_text"].strip()


def test_experiment(experiment_id: str):
    """Test the base model and finetuned model"""
    print("\nStep 1: Load questions")
    # Load questions
    with open("questions.json") as f:
        data = json.load(f)
        questions = data["questions"]

    base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    print(f"\nStep 2: Test the base model : {base_model_id}")
    base_model = BaseInference(model_id=base_model_id)
    experiment_results = {}
    for item in questions:
        base_model_answer = base_model.generate(item["question"])
        experiment_results[item["id"]] = {
            "id": item["id"],
            "question": item["question"],
            "base_model": base_model_answer,
            "fine-tuned_model": "",
        }

    del base_model
    # clear cuda cache
    torch.cuda.empty_cache()

    adapter_path = f"output/Meta-Llama-3.1-8B-Instruct_finetune_{experiment_id}"
    finetuned_model = FinetunedModelInference(base_model_name=base_model_id, adapter_path=adapter_path)

    print(f"\nStep 3: Test the fine-tuned model : {adapter_path}")
    for item in questions:
        finetuned_model_answer = finetuned_model.generate(item["question"])
        experiment_results[item["id"]]["fine-tuned_model"] = finetuned_model_answer

    #  Save results of experiment in results.json by appending the dict results to the existing results.json
    with open("results.json") as f:
        results_data = json.load(f)
        results_data["data"][experiment_id] = experiment_results

    # Save the updated results
    with open("results.json", "w") as f:
        json.dump(results_data, f, indent=4)

    print("\nExperiment results saved to results.json")


# main function
if __name__ == "__main__":
    experiment_id = "20241105_1020"
    test_experiment(experiment_id=experiment_id)
