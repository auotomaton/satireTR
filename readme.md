# Make Satire Boring Again:Reducing Stylistic Bias of Satirical Corpus by Utilizing Generative LLMs

Satire detection is essential for accurately extracting opinions from textual data and combating misinformation online. However, the lack of diverse corpora for satire leads to the problem of stylistic bias which impacts the models' detection performances.
This study proposes a debiasing approach for satire detection, focusing on reducing biases in training data by utilizing generative large language models. 
The approach is evaluated in both cross-domain (irony detection) and cross-lingual (English) settings. Results show that the debiasing method enhances the robustness and generalizability of the models for satire and irony detection tasks in Turkish and English. However, its impact on causal language models, such as Llama-3.1, is limited. Additionally, this work curates and presents the Turkish Satirical News Dataset with detailed human annotations, with case studies on classification, debiasing, and explainability.

## SatireTR: Turkish Satirical News Dataset

Turkish Satirical News Dataset consists of `SATIRICAL` and `NON-SATIRICAL` corpora collected from Turkish satirical news publication Zaytung and Turkish News Agency AA.

The dataset includes 2202 `SATIRICAL` and 4781 `NON-SATIRICAL` articles, and human annotations for 40 of the `SATIRICAL` articles.

### Original Data 

* `satirical_zaytung.csv`
* `nonsatirical_aa.csv`

### Human Annotations

* `satirical_human_annotated_40.docx`
* `satirical_human_annotated_40.pdf`

## Debiasing and Debiased Satirical Data 

This dataset is curated in the scope of the research "Make Satire Boring Again: Reducing Stylistic Bias of Satirical Corpus by Utilizing Generative LLMs". 

The codes and generated data are available under the `Debiasing Pipeline` folder.

## Training Satire Classifiers

First, you need to install libraries as follows:
```
pip install -r requirements.txt
```

Moreover, you need to provide your wandb API key once as follows:
```
wandb.login(key=WANDB_API_KEY)
```

You can evaluate the proposed pipeline on the dataset using the masked langauge models as follows:

```
python debiasing_BERT_based.py \
--model_id "FacebookAI/xlm-roberta-large" \
--train "biased" \ # 'biased/debiased/combined for train'
--cache_dir None \
--skip_train False  \
--wandb_proj_name "zaytung" \ 
```

To evaluate the Llama models, you must first accept the LICENSE AGREEMENT on Hugging Face and generate an access token. Once this is complete, you can run the Llama fine-tuning code as follows:

```
python llama_finetune.py \
--model_id "meta-llama/Llama-3.1-8B-Instruct" \
--hf_token "Hugging Face token" \
--train_file "data/train_combined.csv" \
--test_file "data/onion_test.csv" \
--output_dir "output/biased" \
--cache_dir None \
--skip_train False  \
--epochs 5 \
--batch_size 2 \
--wandb_proj_name "zaytung" \ 
```


# Citation

Please cite the paper as follows if you find the study useful.

```

```

