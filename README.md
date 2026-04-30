# Bank Marketing Response Prediction

ML-проект для предсказания отклика клиента на банковскую маркетинговую кампанию.

## Описание задачи

Цель проекта — построить модель, которая по данным клиента и параметрам маркетинговой кампании оценивает вероятность того, что клиент оформит банковский продукт.

Проект рассматривает задачу не только как бинарную классификацию, но и как задачу **приоритизации клиентов**: банк может не обращаться ко всем клиентам подряд, а выбирать группы с наибольшей вероятностью отклика.

## Dataset

Использован датасет **Bank Marketing**.

Целевая переменная:

```text
y = no  — клиент не оформил депозит
y = yes — клиент оформил депозит
```

В проекте целевая переменная была преобразована:

```text
no  → 0
yes → 1
```

## Features

В датасете есть клиентские признаки и параметры маркетингового контакта:

```text
age, job, marital, education, balance, housing, loan,
contact, day, month, campaign, pdays, previous, poutcome
```

Признак `duration` был исключён из модели, так как он известен только после контакта с клиентом.  
Если использовать его для предсказания до звонка, это приведёт к data leakage.

## Project pipeline

1. Exploratory Data Analysis
2. Target encoding
3. Removing leakage-prone feature `duration`
4. Train/test split with stratification
5. Preprocessing:
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
6. Model training:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
7. Model comparison
8. Top-k targeting analysis
9. Business interpretation

## Target distribution

Задача является несбалансированной: клиентов, которые оформили депозит, заметно меньше, чем клиентов без отклика.

Поэтому в проекте используются метрики:

- ROC-AUC;
- PR-AUC;
- precision;
- recall;
- F1-score;
- top-k response rate;
- lift.

## Model comparison

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.772 | 0.409 | 0.266 | 0.624 | 0.373 |
| Random Forest | 0.793 | 0.432 | 0.370 | 0.564 | 0.447 |
| Gradient Boosting | 0.799 | 0.451 | 0.670 | 0.190 | 0.296 |

Random Forest показал лучший F1-score, однако основная бизнес-задача проекта — не только бинарная классификация, а ранжирование клиентов по вероятности отклика.

Gradient Boosting показал лучшие значения ROC-AUC и PR-AUC, поэтому был выбран как финальная модель для top-k targeting.

## Top-k targeting

Top-k targeting — это подход, при котором банк выбирает клиентов с наибольшей предсказанной вероятностью отклика.

Вместо того чтобы обращаться ко всем клиентам, можно выбрать, например, top-5%, top-10% или top-20% клиентов по вероятности отклика.

| Top-k clients | Clients selected | Response rate | Baseline response rate | Lift | Positive responses |
|---:|---:|---:|---:|---:|---:|
| 5% | 452 | 0.619 | 0.117 | 5.295 | 280 |
| 10% | 904 | 0.512 | 0.117 | 4.378 | 463 |
| 20% | 1808 | 0.357 | 0.117 | 3.049 | 645 |
| 30% | 2712 | 0.276 | 0.117 | 2.357 | 748 |

Baseline response rate на тестовой выборке составляет около **11.7%**.

В top-10% клиентов, выбранных моделью, response rate достигает **51.2%**, что даёт lift **4.38** по сравнению со случайным выбором клиентов.

## Results

Финальная модель — **Gradient Boosting**.

Модель позволяет:

- предсказывать вероятность отклика клиента;
- ранжировать клиентов по вероятности оформления продукта;
- выбирать top-k наиболее перспективных клиентов;
- повышать эффективность маркетинговой кампании за счёт фокусировки на клиентах с высоким response probability.

## Project structure

```text
bank-marketing-response-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda_modeling.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison_pr_auc.png
│   │   ├── top_k_response_rate.png
│   │   └── top_k_lift.png
│   ├── model_comparison.csv
│   └── top_k_analysis.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Tech stack

```text
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Put dataset into:

```text
data/raw/bank-full.csv
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_eda_modeling.ipynb
```

## Future improvements

- Add CatBoost / LightGBM model comparison;
- Add SHAP-based model interpretation;
- Add Streamlit demo for client response prediction;
- Move training and inference logic from notebook to reusable Python scripts.