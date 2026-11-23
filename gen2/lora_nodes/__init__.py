"""Gen2 LoRA utilities."""

from .lora_load import Gen2_LoRALoad
from .lora_save import Gen2_LoRASave

NODE_CLASS_MAPPINGS = {"Gen2_LoRASave": Gen2_LoRASave, "Gen2_LoRALoad": Gen2_LoRALoad}
NODE_DISPLAY_NAME_MAPPINGS = {"Gen2_LoRASave": "Gen2 LoRA Save", "Gen2_LoRALoad": "Gen2 LoRA Load"}
