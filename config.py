from pydantic import BaseModel


class FinetuneConfig(BaseModel):
    # Training parameters
    max_steps: int = 200
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    seed: int = 42
    save_strategy: str = "no"
    optim: str = "adamw_8bit"

    gradient_checkpointing: bool = True
    logging_steps: int = 15

    # LoRA parameters
    r: int = 8
