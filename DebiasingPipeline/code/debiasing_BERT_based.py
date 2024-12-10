from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import torch
import wandb
from tqdm import tqdm 
import argparse

#model_id = "google-bert/bert-base-multilingual-cased"
#model_id = "dbmdz/bert-base-turkish-cased"
#model_id = "FacebookAI/xlm-roberta-large"

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='FacebookAI/xlm-roberta-large', type=str, help='model id')
parser.add_argument('--train', default='biased', type=str, help='biased/debiased/combined for train')
parser.add_argument('--cache_dir', default=None, type=str, help='model cache dir')
parser.add_argument('--skip_train', default=False, type=bool, help='skip to inference')
parser.add_argument('--wandb_proj_name', default="zaytung", type=str, help='wandb project name')
args = parser.parse_args()

model_id = args.model_id
model_name = model_id.split("/")[-1]
cache_dir = args.cache_dir
skip_train = args.skip_train
train_set = args.train
output_dir = "sweep/" + model_name
wandb_proj_name = args.wandb_proj_name

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
epoch = 2

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc, "f1-macro": f1_macro}

if train_set == 'combined':
    combined_train = pd.read_csv("data/train_combined.csv", sep=",", encoding="utf-8", engine="python")
    combined_train = combined_train.rename(columns={"label": "class"})
    combined_train["label"] = combined_train.apply(
        lambda row: (
            0 if row["class"] == "non-satiric" else 1 if row["class"] == "satiric" else None
        ),
        axis=1,
    )
    df = combined_train[["text", "label"]]

else:
    base_train = pd.read_csv("data/train.csv", sep=",", encoding="utf-8", engine="python")
    base_train = base_train.rename(columns={"label": "class"})
    base_train["label"] = base_train.apply(
        lambda row: (
            0 if row["class"] == "non-satiric" else 1 if row["class"] == "satiric" else None
        ),
        axis=1,
    )
    if train_set == 'debiased':
        base_train = base_train.rename(columns={"generated_content": "text"})
    else:
        base_train = base_train.rename(columns={"init_content": "text"})
    df = base_train[["text", "label"]]

#df['label'] = df['label'].astype(int)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# create train and validation splits
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# create three test sets
# zaytung
test_z = pd.read_csv("data/test.csv", sep=",", encoding="utf-8", engine="python")
test_z = test_z.rename(columns={"label": "class"})
test_z["label"] = test_z.apply(
        lambda row: (
            0 if row["class"] == "non-satiric" else 1 if row["class"] == "satiric" else None
        ),
        axis=1,
    )
test_df = test_z[["text", "label"]]
test_df.dropna(inplace=True)
test_df.reset_index(drop=True, inplace=True)

#onion 
test_df_2 = pd.read_csv("data/onion_test.csv", sep=",", encoding="utf-8", engine="python")
test_df_2 = test_df_2[["text", "label"]]
test_df_2.dropna(inplace=True)
test_df_2.reset_index(drop=True, inplace=True)

#ironytr 
test_df_3 = pd.read_csv("data/ironytr_test.csv", sep=",", encoding="utf-8", engine="python")
test_df_3 = test_df_3[["text", "label"]]
test_df_3.dropna(inplace=True)
test_df_3.reset_index(drop=True, inplace=True)

print(f"Train size: {train_df.shape[0]}, Val size: {val_df.shape[0]}, Test size: {test_df.shape[0]} ")
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    #"zaytung": Dataset.from_pandas(test_df),
    #"onion": Dataset.from_pandas(test_df_2),
    #"ironytr": Dataset.from_pandas(test_df_3),
    })
try:
    dataset = dataset.remove_columns(["__index_level_0__"])
except: 
    print("index column does not exist in the dataset!")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
tokenized_dataset = dataset.map(preprocess_function, batched=True).remove_columns(["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "non-satiric", 1: "satiric"}
label2id = {"non-satiric": 0, "satiric": 1}

def init_model(config=None):

    # initialize a new wandb run
    with wandb.init(config=config, project=wandb_proj_name) as run:

        # if called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        run_name = f"model_id:{model_name}|batch:{config.batch_size}|lr:{config.learning_rate}"
        params = {
            "batch_size": config.batch_size, 
            "lr": config.learning_rate, 
            "run_name" : run_name
            }
        
        run.name = run_name
        train(params)
        

def train(params):
    global sweep_no
    sweep_no += 1
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=id2label, label2id=label2id, cache_dir=cache_dir, device_map="cuda:0"
    )

    for p in model.parameters(): p.data = p.data.contiguous()

    training_args = TrainingArguments(
        output_dir=output_dir + "_" + str(sweep_no),
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=params["lr"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=epoch,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=0.5,
        eval_steps=0.5,
        report_to="wandb",
        run_name=params["run_name"],
        load_best_model_at_end=True,
        seed=42,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    evaluate(trainer.model, params["run_name"], "zaytung")
    evaluate(trainer.model, params["run_name"], "onion")
    evaluate(trainer.model, params["run_name"], "ironytr")


def evaluate(model, run_name, test_set):
    global result_dict
    global test_df, test_df_2, test_df_3
    if test_set == "onion": 
        test_df = test_df_2
    elif test_set == "ironytr":
        test_df = test_df_3

    references = test_df["label"].to_list()
    texts = test_df["text"].to_list()
    predictions = []
    diff = {}

    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        y_pred = logits.argmax().item()
        predictions.append(y_pred)

    print(classification_report(references, predictions, labels=[0,1]))

    conf_matrix = str(confusion_matrix(references, predictions).tolist())
    metrics = {
        "f1-macro" : f1_score(references, predictions, average='macro'),
        "f1-micro" : f1_score(references, predictions, average='micro'),
        "f1-weighted" : f1_score(references, predictions, average='weighted'),
        "precision" : precision_score(references, predictions, average='weighted'),
        "recall" : recall_score(references, predictions, average='weighted'),
        "accuracy" : accuracy_score(references, predictions)
    }

    result_dict["confusion matrix"].append(conf_matrix)
    result_dict["metrics"].append(metrics)
    result_dict["run name"].append((run_name + "-" + test_set))
    result_dict["f1-macro"].append(metrics["f1-macro"])


result_dict = {"run name":[], "metrics": [], "confusion matrix": [], "f1-macro":[]}
sweep_no = 0

if skip_train:
    model = AutoModelForSequenceClassification.from_pretrained(output_dir + "_" + str(1), num_labels=2, id2label=id2label, label2id=label2id, cache_dir=cache_dir)
    evaluate(model, run_name=None)

else:
    sweep_config = {
        'method': 'grid', 
        'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values': [5e-5, 2e-5, 1e-4]
                },
            'batch_size': {
                'values': [8, 16]
                },
        }
    }

    # wandb.login(key=bb1d16d7e0ec95e5bf337e52d876ed48c1d773df)
    sweep_id = wandb.sweep(sweep_config, project=wandb_proj_name)

    wandb.agent(sweep_id, function=init_model, count=4)
    
    result_df = pd.DataFrame.from_dict(result_dict)
    print(result_df)
    result_df.to_csv("sweep_" + train_set + "_" + model_name + ".csv", sep="\t", encoding="utf-8",  mode='a', header=False)

    