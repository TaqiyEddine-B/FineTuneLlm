import torch
from pydantic import BaseModel, ConfigDict


class FinetuneConfig(BaseModel):
    # Training parameters
    max_steps: int = 80
    per_device_train_batch_size: int = 2
    learning_rate : float = 5e-5
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    seed: int = 42
    save_strategy: str = "no"
    optim: str = "adamw_8bit"

    gradient_checkpointing: bool = True
    logging_steps: int = 15

    # LoRA parameters
    r: int = 8


class InferenceConfig(BaseModel):
    # Inference parameters
    model_config = ConfigDict(arbitrary_types_allowed=True)
    torch_dtype: torch.dtype = torch.float32  # torch.float16
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
