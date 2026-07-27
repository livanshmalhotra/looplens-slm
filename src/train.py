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
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Optional import for trl SFTTrainer
try:
    from trl import SFTTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

def load_yaml_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_instruction_text(example: dict, tokenizer: AutoTokenizer) -> str:
    """Formats instruction, input, and output into chat template format."""
    system_prompt = example.get(
        "instruction",
        "You are LoopLens AI, a procurement analytics assistant. Always respond in clear, non-technical language."
    )
    user_input = example.get("input", "")
    response = example.get("output", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    # Standard Qwen2.5 chat template format fallback
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

def require_cuda():
    """Hard-fails if no NVIDIA GPU is visible to torch, instead of silently
    falling back to CPU training."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[FATAL] No CUDA-capable NVIDIA GPU detected by torch.cuda.is_available().\n"
            "This script is configured to train on GPU only. Checklist:\n"
            "  1. Confirm the machine actually has an NVIDIA GPU: run `nvidia-smi`.\n"
            "  2. Confirm your PyTorch build has CUDA support: "
            "`python -c \"import torch; print(torch.__version__, torch.version.cuda)\"` "
            "should NOT print 'None' for cuda.\n"
            "  3. If you installed the CPU-only torch wheel, reinstall a CUDA build, e.g.:\n"
            "     pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  4. If running in Docker, make sure the container was started with --gpus all "
            "and nvidia-container-toolkit is installed on the host.\n"
        )
    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[*] CUDA is available. Found {device_count} GPU(s). Using device 0: {gpu_name}")

def train(config_path: str):
    """Executes QLoRA SFT Training Loop for Qwen2.5-0.5B-Instruct on NVIDIA GPU."""
    require_cuda()

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

    base_model_name = model_cfg["base_model_name_or_path"]
    print(f"[*] Loading Base Model Tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Configuring 4-bit Quantization (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True)
    )
    print(f"[*] Loading Base Model: {base_model_name} on CUDA...")
    # NOTE: no silent try/except fallback here anymore. If quantized loading
    # fails, the script raises the real error instead of quietly retrying on CPU.
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

    # Build TrainingArguments parameters dynamically for transformers compatibility
    kwargs = {
        "output_dir": train_cfg["output_dir"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "logging_steps": train_cfg["logging_steps"],
        "eval_steps": train_cfg["eval_steps"],
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", False),
        "report_to": "none",
        "bf16": train_cfg.get("bf16", False),
        "fp16": train_cfg.get("fp16", False),
        "optim": train_cfg.get("optim", "paged_adamw_8bit"),
    }

    # Handle eval strategy parameter name across transformers versions
    eval_strat = train_cfg.get("evaluation_strategy", train_cfg.get("eval_strategy", "steps"))
    if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        kwargs["eval_strategy"] = eval_strat
    else:
        kwargs["evaluation_strategy"] = eval_strat

    training_args = TrainingArguments(**kwargs)

    max_seq_length = data_cfg.get("max_seq_length", 1024)

    if HAS_TRL:
        print("[*] Using TRL SFTTrainer...")
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
            formatting_func=formatting_prompts_func,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args
        )
    else:
        print("[*] Using standard HuggingFace Trainer...")
        def preprocess_function(examples):
            texts = []
            for i in range(len(examples["instruction"])):
                ex = {
                    "instruction": examples["instruction"][i],
                    "input": examples["input"][i],
                    "output": examples["output"][i]
                }
                texts.append(format_instruction_text(ex, tokenizer))
            # Return tokenized dictionary (DataCollatorForLanguageModeling handles labels creation & dynamic padding)
            return tokenizer(texts, max_length=max_seq_length, truncation=True)

        tokenized_train = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_val = dataset["validation"].map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": tokenized_train,
            "eval_dataset": tokenized_val,
            "data_collator": data_collator
        }
        if "processing_class" in Trainer.__init__.__code__.co_varnames:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in Trainer.__init__.__code__.co_varnames:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

    print(f"[*] Starting Fine-Tuning Training Loop for LoopLens SLM on GPU...")
    trainer.train()

    final_adapter_path = os.path.join(train_cfg["output_dir"], "final_adapter")
    os.makedirs(final_adapter_path, exist_ok=True)
    print(f"[OK] Fine-Tuning complete! Saving LoRA adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoopLens QLoRA Trainer (GPU-only)")
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)