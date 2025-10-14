"""
Causal LLM Model with Intervention-Based Security

This module implements the core causal intervention model for LLM security.
Based on the theoretical framework in theory/causal_formalization.md
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple


class CausalLLMModel(nn.Module):
    """
    LLM with causal intervention for security.

    Implements the causal graph:
        S (System Instruction) → R (Representation) ← U (User Input)
                                        ↓
                                    O (Output)

    Goal: Ensure P(O | do(S), U) is invariant to instruction-bearing changes in U
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_4bit: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the causal LLM model with LoRA.

        Args:
            model_name: HuggingFace model identifier
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout rate for LoRA layers
            load_in_4bit: Use 4-bit quantization for memory efficiency
            device_map: Device placement strategy
        """
        super().__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set up quantization config for 4-bit if requested
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)

        # Additional projection layer for causal representation learning
        hidden_size = self.base_model.config.hidden_size
        self.causal_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_representation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_representation: If True, return intermediate representation R

        Returns:
            Dictionary containing:
                - logits: Output logits [batch_size, seq_len, vocab_size]
                - representation: (optional) Causal representation R
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        result = {"logits": outputs.logits}

        if return_representation:
            # Extract last layer hidden states
            hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

            # Pool to get representation (use last token or mean pooling)
            if attention_mask is not None:
                # Mean pooling with attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                # Simple mean pooling
                pooled = hidden_states.mean(dim=1)

            # Apply causal projection
            representation = self.causal_projection(pooled)
            result["representation"] = representation

        return result

    def generate(
        self,
        system_instruction: str,
        user_input: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate output given system instruction and user input.

        Args:
            system_instruction: System-level instruction (S)
            user_input: User input (U)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated text output (O)
        """
        # Format prompt
        prompt = self._format_prompt(system_instruction, user_input)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def _format_prompt(self, system_instruction: str, user_input: str) -> str:
        """
        Format the prompt combining system instruction and user input.

        Args:
            system_instruction: System instruction (S)
            user_input: User input (U)

        Returns:
            Formatted prompt string
        """
        # Standard instruction format
        prompt = f"""### System Instruction:
{system_instruction}

### User Input:
{user_input}

### Response:
"""
        return prompt

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: str):
        """Save the model (LoRA weights only)."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save causal projection separately
        torch.save(
            self.causal_projection.state_dict(),
            f"{save_directory}/causal_projection.pt"
        )

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load a saved model."""
        model = cls(**kwargs)

        # Load LoRA weights
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(
            model.base_model,
            load_directory
        )

        # Load causal projection
        model.causal_projection.load_state_dict(
            torch.load(f"{load_directory}/causal_projection.pt")
        )

        return model


if __name__ == "__main__":
    # Example usage
    print("Initializing Causal LLM Model...")
    model = CausalLLMModel(
        model_name="meta-llama/Llama-2-7b-hf",
        lora_r=16,
        load_in_4bit=True
    )

    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    print("Model initialized successfully!")
