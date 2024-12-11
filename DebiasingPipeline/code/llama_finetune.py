import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments, pipeline
import wandb
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import login
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import argparse


parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='meta-llama/Llama-3.1-8B-Instruct', type=str, help='model id')
parser.add_argument('--hf_token', default='HF_TOKEN', type=str, help='Hugging Face token')
parser.add_argument('--train_file', default='data/train_combined.csv', type=str, help='biased/debiased/combined for train')
parser.add_argument('--train_file', default='data/onion_test.csv', type=str, help='biased/debiased/combined for test')
parser.add_argument('--output_dir', default="output/biased", type=str, help='model cache dir')
parser.add_argument('--cache_dir', default=None, type=str, help='model cache dir')
parser.add_argument('--skip_train', default=False, type=bool, help='skip to inference')
parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--wandb_proj_name', default="ironytr", type=str, help='wandb project name')
args = parser.parse_args()

login(token=args.hf_token)

# Hugging Face model id
model_id = args.model_id
train_file = args.train_file
test_file = args.test_file
OUTPUT_DIR = args.output_dir
skip_train = args.skip_train

label_map = {"satiric": 1, "satirik":1, "ironik":1, "ironic":1, "non-satiric": 0, "non-ironic": 0}
system_message = """You are an AI agent. You are asked to classify the given text as 'satiric' or 'non-satiric'. You must classify the following text as 'satiric' or 'non-satiric'."""


def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    #outputs = pipe(prompt, max_new_tokens=64, do_sample=True, temperature=0.001, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    outputs = pipe(prompt, num_beams=5, max_new_tokens=50, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    return predicted_answer.lower(), sample["messages"][2]["content"].lower()

def replace_labels(sample):
    sample['label'] = 'satiric' if sample['label'] == 1 else 'non-satiric'
    return sample
     
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": sample["text"]},
      {"role": "assistant", "content": sample["label"]}
    ]
  }

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings
#model.pad_token_id = tokenizer.pad_token_id
#model.config.pad_token_id = tokenizer.pad_token_id

if skip_train:
    print(f"INFERENCE MODE: {test_file}, MODEL_DIR: {OUTPUT_DIR}")
    test_dataset = load_dataset('csv', data_files=test_file, delimiter=',', split="train")
    test_dataset = test_dataset.map(replace_labels)
    test_dataset = test_dataset.map(create_conversation, remove_columns=test_dataset.features,batched=False)
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        #low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    #merged_model.push_to_hub(hub_id)
    #tokenizer.push_to_hub(hub_id)

    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer, device="cuda")

    predictions = []
    targets = []
    number_of_eval_samples = len(test_dataset)
    # iterate over eval dataset and predict
    for s in tqdm(test_dataset.shuffle().select(range(number_of_eval_samples))):
        pred, label = evaluate(s)

        try:
            predictions.append(label_map[pred])
        except:
            if "non-satiric" in pred:
                predictions.append(0)
            elif "satiric" in pred or "satirik" in pred or "ironic" in pred or "ironik" in pred:
                predictions.append(1)
            else:
                print(f"Wrong prediction: {pred}")
                continue
        
        targets.append(label_map[label])
        
    metrics = {
        "f1-macro" : f1_score(targets, predictions, average='macro'),
        "f1-micro" : f1_score(targets, predictions, average='micro'),
        "f1-weighted" : f1_score(targets, predictions, average='weighted'),
        "precision" : precision_score(targets, predictions, average='weighted'),
        "recall" : recall_score(targets, predictions, average='weighted'),
        "accuracy" : accuracy_score(targets, predictions)
    }
    print(metrics)

else:
    dataset = load_dataset('csv', data_files=train_file, delimiter=',', split="train")
    dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
    #dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        cache_dir=args.cache_dir,
        attn_implementation="flash_attention_2",
        quantization_config=None if skip_train else bnb_config,
    )
    model.config.use_cache = False


    # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
        task_type="CAUSAL_LM",
    )


    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model_name = model_id.split("/")[-1]
    exp_name = OUTPUT_DIR.split("/")[-1]
    WANDB_PROJECT = f"""ft-{model_name}-{exp_name}"""
    wandb.init(project=args.wandb_proj_name, name=WANDB_PROJECT)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR, # directory to save and repository id
        num_train_epochs=args.epoch,                     # number of training epochs
        per_device_train_batch_size=args.batch_size,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=1,                       # log every 10 steps
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=0.2,
        learning_rate=5e-5,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        fp16=False,
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        #hub_model_id=hub_id,
        report_to="wandb",                # report metrics to tensorboard
        run_name=WANDB_PROJECT,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


    max_seq_length = 2048 # max sequence length for model and packing of the dataset
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        #peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        #dataset_kwargs={
        #    "add_special_tokens": False,  # We template with special tokens
        #    "append_concat_token": False, # No need to add additional separator token
        #}
    )

    trainer.train()
    trainer.save_model()
