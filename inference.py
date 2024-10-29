import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Inference:
    def __init__(self, base_model_name: str, adapter_path: str):
        print(f"Loading base model {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            # trust_remote_code=True
        )

        # Load and apply the LoRA adapter
        print(f"Loading adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt, max_new_tokens=50, temperature=0.1, top_p=0.9):
        torch.cuda.empty_cache()  # Clear GPU cache before generation

        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>
        You are Question/answer assistant.
        Your answer must begin with capital letter and end with full stop.
        <|end_header_id|>{prompt}<|eot_id|>
        """
        # inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        # outputs = self.model.generate(
        #     **inputs,
        #     max_length=max_length,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     num_return_sequences=1,
        #     # clean_up_tokenization_spaces=True,  # Clean output
        #     # return_full_text=False  # Only return new generated text
        # )
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True,  # Clean output
            return_full_text=False,  # Only return new generated text
        )
        return result[0]["generated_text"].strip()

        # return response


class HuggingfaceInference:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with specific GPU configurations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use float16 for GPU memory efficiency
            device_map="cuda:0",  # Explicitly use first GPU
        )

        # Create pipeline with GPU specifications
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt, max_new_tokens=50, temperature=0.1, top_p=0.9):
        torch.cuda.empty_cache()  # Clear GPU cache before generation
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>
        You are Question/answer assistant.
        Your answer must begin with capital letter and end with full stop.
        <|end_header_id|>{prompt}<|eot_id|>
        """

        result = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            clean_up_tokenization_spaces=True,  # Clean output
            return_full_text=False,  # Only return new generated text
        )
        return result[0]["generated_text"].strip()


# main function
if __name__ == "__main__":
    # base_model = "NousResearch/Meta-Llama-3.1-8B"
    # base_model = "meta-llama/Meta-Llama-3.1-8B"
    base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    adapter_path = "output/Meta-Llama-3.1-8B_finetune_05_23_29102024"
    inference = Inference(base_model_name=base_model, adapter_path=adapter_path)
    print(inference.generate("What is the capital of France?"))

    # Example with direct Huggingface model
    # hf_inference = HuggingfaceInference(model_id=base_model)
    # print(hf_inference.generate("What is the capital of France?"))
