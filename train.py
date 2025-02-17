from dataclasses import dataclass, field
from typing import Optional, List
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="processed_data.json"
    )
    dataset_config: Optional[str] = field(default=None)

@dataclass
class LoraArguments:
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    bias: str = field(default="none")
    task_type: str = field(default="CAUSAL_LM")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="TinyLlama-1.1B-qlora-quantization")
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use"})
    max_seq_length: int = field(default=2048)
    overwrite_output_dir: bool = field(default=True)
    num_train_epochs: int = field(default=1)
@dataclass
class QuantizationArguments:
    use_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="float16")
    bnb_4bit_use_double_quant: bool = field(default=True)

def format_prompt(sample):
    prompt = """<|user|>
You are a math expert. Please answer the following question by providing a detailed, step-by-step explanation, as if you were explaining to a 6-year-old. All answer must be in Vietnamese. Use clear, simple Vietnamese and appropriate mathematical notation.

A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and final answer must be enclosed within <think> and <answer> tags respectively.
User: {question}
<|assistant|>
{answer}"""
    return prompt.format(
        question=sample['query_vi'],
        answer=sample['reasoning']
    )

def train():
    from dataset_loading_fix import load_training_data

    # If using Colab or Kaggle:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, TrainingArguments, QuantizationArguments)
    )
    # model_args, data_args, lora_args, train_args, quant_args = parser.parse_dict({})
    # Local
    model_args, data_args, lora_args, train_args, quant_args = parser.parse_args_into_dataclasses()

    # Load dataset
    data = load_training_data(data_args.data_path)

    if isinstance(data, dict):
        data = data["train"]
    data = data.shuffle(seed=42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side='right'
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = train_args.max_seq_length

    # BitsAndBytes quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_args.use_4bit,
        bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant_args.bnb_4bit_use_double_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=lora_args.task_type,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=data,
        tokenizer=tokenizer,
        formatting_func=format_prompt,
    )

    # Train and save
    trainer.train()
    model.save_pretrained(train_args.output_dir)

if __name__ == "__main__":
    train()
