import os, random, numpy as np, torch, datetime, re, wandb
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from .aes_dataloader import LlamaAESCollator
from .number_tokenizer import AutoNumberTokenizer
from .custom_trainer import CustomTrainer

# -------------------------------------------------
# Utils
# -------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 완전 재현성 (성능 약간 저하)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)

def make_unique_output_dir(baseline: bool, ratio: float, ntl_weight: float, loss_type: str) -> str:
    tag = "baseline" if baseline else f"ntl_{ntl_weight}"
    ratio_tag = f"ratio_{ratio}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"./runs/{sanitize_name(tag)}_{sanitize_name(ratio_tag)}_{timestamp}_{loss_type}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def init_wandb(baseline: bool, ratio: float, ntl_weight: float, output_dir: str):
    project_name = sanitize_name(f"llama3_AES_{'baseline' if baseline else 'NTL'}")
    run_name = sanitize_name(f"{'baseline' if baseline else f'ntl_{ntl_weight}'}_r{ratio}_{datetime.datetime.now().strftime('%m%d_%H%M')}")
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_RUN_NAME"] = run_name
    wandb.init(project=project_name, name=run_name, dir=output_dir, reinit=True)
    print(f"WandB initialized → Project: {project_name}, Run: {run_name}")

def sample_ratio(dataset, ratio):
    if ratio < 1.0:
        n = max(1, int(len(dataset) * ratio))
        dataset = dataset.select(range(n))
    return dataset

# -------------------------------------------------
# Train function
# -------------------------------------------------
def train_model(baseline: bool = False, ntl_weight: float = 2.0, ratio: float = 1.0, loss_type: str = "mse"):
    set_seed(42)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = make_unique_output_dir(baseline, ratio, ntl_weight, loss_type)
    max_seq_length = 2800
    use_bf16 = torch.cuda.is_bf16_supported()

    init_wandb(baseline, ratio, ntl_weight, output_dir)

    # Dataset load (train, valid, test 동일한 비율)
    train_ds = sample_ratio(load_dataset('json', data_files="./aes_dataset/train.jsonl")['train'], ratio)
    valid_ds = sample_ratio(load_dataset('json', data_files="./aes_dataset/valid.jsonl")['train'], ratio)
    test_ds  = sample_ratio(load_dataset('json', data_files="./aes_dataset/test.jsonl")['train'], ratio)

    tokenizer = AutoNumberTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = LlamaAESCollator(tokenizer, max_seq_length=max_seq_length)

    # Model & LoRA config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))

    # Trainer args
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        bf16=use_bf16,
        fp16=(not use_bf16),
        save_total_limit=2,
        report_to=["wandb"],
        load_best_model_at_end=True,   # ✅ Best checkpoint 자동 로드
        greater_is_better=False,      # ✅ Loss 기준으로 best 모델 선택
        seed=42,
        remove_unused_columns=False,   # ✅ collator 문제 방지
    )

    if baseline:
        ntl_weight = 0.0
    
    trainer = CustomTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        num_tokenizer=tokenizer,
        order_numbers=True,
        ntl_weight=ntl_weight,
        loss_type=loss_type,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    wandb.finish()

    # ✅ Best model already loaded (no reloading needed)
    print(f"Training complete. Best model already loaded into memory.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, test_ds, tokenizer, output_dir
