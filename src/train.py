import os
import sys
import yaml
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def load_yaml_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_instruction_text(example: dict, tokenizer: AutoTokenizer) -> str:
    """Formats instruction, input, and output into chat template format."""
    system_prompt = example.get("instruction", "You are a specialized Procurement and Spend Analytics AI Assistant.")
    user_input = example.get("input", "")
    response = example.get("output", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ]
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback formatting
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n{response}"

def train(config_path: str):
    """Executes QLoRA SFT Training Loop."""
    cfg = load_yaml_config(config_path)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    print(f"[*] Loading dataset: {data_cfg['train_file']} and {data_cfg['val_file']}")
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["val_file"]
        }
    )

    print(f"[*] Setting up 4-bit Quantization Config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True)
    )

    base_model_name = model_cfg["base_model_name_or_path"]
    print(f"[*] Loading Base Model: {base_model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    print(f"[*] Configuring LoRA Adapters...")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        report_to=train_cfg["report_to"]
    )

    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i]
            }
            output_texts.append(format_instruction_text(example, tokenizer))
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=data_cfg["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args
    )

    print(f"[*] Starting Fine-Tuning Training Loop...")
    trainer.train()

    final_adapter_path = os.path.join(train_cfg["output_dir"], "final_adapter")
    print(f"[OK] Fine-Tuning complete! Saving LoRA adapter to {final_adapter_path}")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoopLens QLoRA Trainer")
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)
