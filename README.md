# BEEP reimplementation


## 0. Data Preparation
### 0. Official Github에서 필요한 모델 및 데이터 가져오기
- [This](https://github.com/allenai/BEEP) repo의 README에 적혀있는, 아래의 데이터를 AWS CLI로 다운받기
- ```aws s3 sync --no-sign-request s3://ai2-s2-beep models/```
- 다운받은거 중에서 ```PMC_id_map.csv```를 data/ 폴더에 옮김. 이게 PMC DB와 PubMED DB 내의 document id mapping을 해주는 파일.


### 1. MIMIC (main task)
- MIMIC3를 [this](https://github.com/bvanaken/clinical-outcome-prediction/tree/master/tasks)를 따라서 전처리하고, ```MIMIC_PREPROCESSED/```에 두기

### 2. PubMED 준비하기 (retrieval DB)
- ```preprocessed/retrieve_abstracts.ipynb``` 를 실행시켜서, task에 맞는 pubmed article을 크롤링하기

0. Data preparation이 끝나면 아래의 데이터들이 준비되어야 함.
```
BEEP_reimplement/
    MIMIC_PREPROCESSED/
        - MP_IN_adm_test.csv  
        - MP_IN_adm_train.csv  
        - MP_IN_adm_val.csv
    data/
        PMC_id_map.csv
        pubmed_db/
            - mortality.pck # set of retrieved ids from pubmed and pmc. integer for pubmed and 'PMC'+integer for pubmed  
            - pubmed_texts_and_dates.mortality.pkl
            - pmc_texts_and_dates.mortality.pkl
```

## 1. Train Dense-retriever and Reranker on TREC2016-CDS (Clinical Decision Support) task

- Dense-retriver / reranker 학습
```python retrieval/train_biencoder.py```
```python retrieval/train_reranker.py```

- 학습된 모델들로 admission note를 위한 retrieval 하기
  - Admission note와 관련있을거같은 pubmed abstracts를 실제로 retrieval하기
```python retrieval/run_dense_retrieval.py```


## 2. Train and test Main model for Clinical Prediction Tasks
### 1. Models w/o retrieval
```python prediction/train_predictor.py```

### 2. Retrieval-augmented predictor
```strategy = <avg,wavg,svote,wvote```
```python prediction/train_predictor.py --task=MP_IN --num_doc_for_augment 5 --augment_strategy avg --predictor_exp_name strategy-avg.doc5```

## 3. Inference the model w/o retrieval
```python prediction/evaluate_predictor.py```

## 3. Inference the retrieval-augmented predictor
```python prediction/evaluate_predictor.py  --task=MP_IN --num_doc_for_augment 5 --augment_strategy avg --predictor_exp_name strategy-avg.doc5``` 