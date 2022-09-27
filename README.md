# BEEP reimplementation


## 0. Data Preparation
### 1. MIMIC (main task)
- MIMIC3를 [this](https://github.com/bvanaken/clinical-outcome-prediction/tree/master/tasks)를 따라서 전처리하고,.
- Locate it in ```MIMIC_PREPROCESSED/``` directory.

### 2. PubMED 준비하기 (retrieval DB)
- ```preprocessed/retrieve_abstracts.ipynb``` 를 실행시켜서, task에 맞는 pubmed article을 검색하기

```
BEEP_reimplement/
    MIMIC_PREPROCESSED/
        - MP_IN_adm_test.csv  
        - MP_IN_adm_train.csv  
        - MP_IN_adm_val.csv
    data/pubmed_db/
        - mortality.pck # set of retrieved ids from pubmed and pmc. integer for pubmed and 'PMC'+integer for pubmed  
        - pubmed_texts_and_dates.mortality.pkl
        - pmc_texts_and_dates.mortality.pkl
```

## 1. Train Bi-encoder and Reranker on TREC2016-CDS (Clinical Decision Support) task

- Retrieval / reranker 학습
```python retrieval/train_biencoder.py```
```python retrieval/train_reranker.py```

- 실제 retrieval 하기
```python retrieval/run_dense_retrieval.py```


## 2. Train Main model for Clinical Prediction Tasks
### 1. Models w/o retrieval
```python prediction/train_predictor.py```

### 2. Retrieval-augmented predictor
```strategy = <avg,wavg,svote,wvote```
```python prediction/train_predictor.py --task=MP_IN --num_doc_for_augment 5 --augment_strategy avg --predictor_exp_name strategy-avg.doc5```

## 3. Inference
```python prediction/evaluate_predictor.py```