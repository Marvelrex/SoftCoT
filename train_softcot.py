import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from fastNLP import logger


from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader
from llm_model import EfficientSoftCoTFromSmallModel
from utils import pre_process_strategy_qa, pre_process_gsm8k, pre_process_aqua, CustomDataCollator
from sft_config import DEFAULT_SFT_CONFIG


args = argparse.ArgumentParser()
args.add_argument('--large_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args.add_argument('--small_model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
args.add_argument('--output_name', type=str, required=True)
args.add_argument('--batch_size', type=int, default=2)
args.add_argument('--task_name', type=str, choices=[
    'gsm8k', 'strategyqa', 'asdiv-aug', 'aqua',
])
args.add_argument('--num_thought_tokens', type=int, default=2)
args.add_argument('--n_epochs', type=float, default=2.0)
args.add_argument('--k_shot', type=int, default=0)
args.add_argument('--learning_rate', type=float, default=1e-4)
args.add_argument('--weight_decay', type=float, default=0.01)
args.add_argument('--grad_accum', type=int, default=8)
args.add_argument('--warmup_ratio', type=float, default=0.03)
args.add_argument('--logging_steps', type=int, default=25)
args.add_argument('--save_steps', type=int, default=250)
args.add_argument('--optim', type=str, default='adamw_torch')
args.add_argument('--lr_scheduler_type', type=str, default='linear')
args.add_argument('--max_grad_norm', type=float, default=1.0)
args.add_argument('--seed', type=int, default=42)
args.add_argument('--max_length', type=int, default=2048)
args.add_argument('--max_samples', type=int, default=None)
args.add_argument('--bf16', dest='bf16', action='store_true')
args.add_argument('--no_bf16', dest='bf16', action='store_false')
args.set_defaults(bf16=True)
args.add_argument('--pad_to_max', dest='pad_to_max', action='store_true')
args.add_argument('--no_pad_to_max', dest='pad_to_max', action='store_false')
args.set_defaults(pad_to_max=False)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
arg = args.parse_args()

logger.info(f'args: {arg.__dict__}')

large_model_id = arg.large_model_id
small_model_id = arg.small_model_id
output_name = arg.output_name
batch_size = arg.batch_size
task_name = arg.task_name
n_epochs = arg.n_epochs
num_thought_tokens = arg.num_thought_tokens
k_shot = arg.k_shot
learning_rate = arg.learning_rate
weight_decay = arg.weight_decay
grad_accum = arg.grad_accum
warmup_ratio = arg.warmup_ratio
logging_steps = arg.logging_steps
save_steps = arg.save_steps
optim = arg.optim
lr_scheduler_type = arg.lr_scheduler_type
max_grad_norm = arg.max_grad_norm
seed = arg.seed
max_length = arg.max_length
max_samples = arg.max_samples
bf16 = arg.bf16
pad_to_max = arg.pad_to_max
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model


large_model_name = large_model_id.split('/')[-1]
small_model_name = small_model_id.split('/')[-1]
post_fix = f'{task_name}-{n_epochs}-{num_thought_tokens}-{large_model_name}-{small_model_name}'
results_root = os.getenv('SOFTCOT_RESULTS_DIR', './results')
logs_root = os.getenv('SOFTCOT_LOGS_DIR', './logs')
ckpt_root = os.getenv('SOFTCOT_CKPT_DIR', './ckpt')
output_dir = os.path.join(results_root, f'{output_name}-{post_fix}')
log_dir = os.path.join(logs_root, f'{output_name}-{post_fix}')
save_model_dir = os.path.join(ckpt_root, f'{output_name}-{post_fix}')

logger.info(f'Output Dir: {output_dir}')
logger.info(f'Log Dir: {log_dir}')
logger.info(f'Save Model Dir: {save_model_dir}')

# Fixed global seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
if hf_token is None:
    logger.warning('HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set; gated models may 401.')
else:
    logger.info(f'HF token detected; prefix={hf_token[:8]}')
    try:
        from huggingface_hub import HfApi
        sha = HfApi(token=hf_token).model_info(small_model_id, timeout=10).sha
        logger.info(f'HF access check ok; sha={sha}')
    except Exception as e:
        logger.error(f'HF access check failed: {e}')

base_tokenizer = AutoTokenizer.from_pretrained(large_model_id, token=hf_token)
assistant_tokenizer = AutoTokenizer.from_pretrained(small_model_id, token=hf_token)

if 'Llama' in large_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in large_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in small_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in small_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

if tune_assistant_model or tune_base_model:
    logger.info(
        f"Applying baseline SFT LoRA config "
        f"(use_lora={DEFAULT_SFT_CONFIG.use_lora}, "
        f"r={DEFAULT_SFT_CONFIG.lora_r}, "
        f"alpha={DEFAULT_SFT_CONFIG.lora_alpha}, "
        f"dropout={DEFAULT_SFT_CONFIG.lora_dropout}, "
        f"targets={DEFAULT_SFT_CONFIG.lora_target_modules})."
    )

model = EfficientSoftCoTFromSmallModel(
    small_model_id,
    large_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
    lora_config=DEFAULT_SFT_CONFIG,
)

logger.info(f'Successfully Init Model `{model.__class__.__name__}`')

trainable_param = 0
total_param = 0
for n, p in model.named_parameters():
    if p.requires_grad:
        trainable_param += p.view(-1).size(0)
    total_param += p.view(-1).size(0)
logger.info(f'Trainable Parameters: {trainable_param}; Total Parameters: {total_param}')

if task_name in ['gsm8k']:
    db = GSM8KLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    preprocess_method = pre_process_aqua
else:
    raise NotImplementedError

train_dataset = db.get_dataset('train')
eval_dataset = db.get_dataset('dev')

if k_shot > 0:
    train_dataset = train_dataset[: k_shot]
if max_samples is not None:
    train_dataset = train_dataset[: max_samples]

train_rows = []
for ins in tqdm(train_dataset, desc='Preprocess Training Set'):
    train_rows.append(
        preprocess_method(
            ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
            add_bot_eot=True, split='train',
            base_special_token=base_special_token,
            assistant_special_token=assistant_special_token,
            base_backbone=base_backbone,
            assistant_backbone=assistant_backbone,
            max_len=max_length,
        )
    )

eval_rows = []
for ins in tqdm(eval_dataset, desc='Preprocess Testing Set'):
    eval_rows.append(
        preprocess_method(
            ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
            add_bot_eot=True, split='dev',
            base_special_token=base_special_token,
            assistant_special_token=assistant_special_token,
            base_backbone=base_backbone,
            assistant_backbone=assistant_backbone,
            max_len=max_length,
        )
    )

train_data = Dataset.from_pandas(pd.DataFrame(train_rows))
eval_data = Dataset.from_pandas(pd.DataFrame(eval_rows))

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    eval_strategy='epoch',
    save_strategy='steps',
    save_steps=save_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    warmup_ratio=warmup_ratio,
    num_train_epochs=n_epochs,
    optim=optim,
    bf16=bf16,
    lr_scheduler_type=lr_scheduler_type,
    max_grad_norm=max_grad_norm,
    logging_dir=log_dir,
    logging_steps=logging_steps,
    remove_unused_columns=True,
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=CustomDataCollator(
        pad_to_max=pad_to_max,
        max_length=max_length,
    ),
)
trainer.train()

model.save_pretrained(save_model_dir)
logger.info(f'Finish training, save model to dir `{save_model_dir}`')


