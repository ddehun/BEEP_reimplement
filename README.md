# BEEP reimplementation


## 0. Data Preparation
### 1. MIMIC (main task)
- Preprocess MIMIC3 following [this](https://github.com/bvanaken/clinical-outcome-prediction/tree/master/tasks) Github repository.
 
### 2. PubMED (retrieval DB)
- entrez_outcome_specific_retrieval.ipynb 실행


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


## 2. Train Main model for Clinical Prediction Tasks


