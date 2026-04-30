# Bank Marketing Response Prediction

ML-проект для предсказания отклика клиента на банковскую маркетинговую кампанию.

Проект решает задачу бинарной классификации: по данным клиента и параметрам маркетингового контакта нужно предсказать, оформит ли клиент банковский продукт. Помимо обычного предсказания `yes / no`, проект рассматривает задачу с бизнес-точки зрения: модель используется для ранжирования клиентов по вероятности отклика, чтобы банк мог выбирать наиболее перспективных клиентов для коммуникации.

---

## Описание задачи

Банк проводит маркетинговую кампанию и предлагает клиентам оформить срочный депозит.  
Задача — построить модель, которая по историческим данным сможет оценить вероятность того, что клиент согласится на предложение.

В реальной ситуации банк обычно не хочет обращаться ко всем клиентам подряд: это дорого, занимает время операторов и может ухудшать клиентский опыт. Поэтому полезнее не просто предсказывать класс `yes / no`, а ранжировать клиентов по вероятности отклика и выбирать top-k наиболее перспективных клиентов.

В проекте решаются две связанные задачи:

1. **Binary classification** — предсказать, откликнется клиент или нет.
2. **Top-k targeting** — выбрать клиентов с наибольшей вероятностью отклика для приоритетной коммуникации.

---

## Dataset

Использован датасет **Bank Marketing**.

Каждая строка в датасете описывает клиента и параметры контакта в рамках маркетинговой кампании банка.

Целевая переменная:

```text
y = no  — клиент не оформил депозит
y = yes — клиент оформил депозит
```

В проекте целевая переменная была преобразована в числовой формат:

```text
no  → 0
yes → 1
```

---

## Features

В датасете используются клиентские признаки и параметры маркетингового контакта.

Примеры признаков:

```text
age        — возраст клиента
job        — тип занятости
marital    — семейное положение
education  — уровень образования
balance    — баланс клиента
housing    — наличие жилищного кредита
loan       — наличие персонального кредита
contact    — тип контакта
day        — день последнего контакта
month      — месяц последнего контакта
campaign   — количество контактов в текущей кампании
pdays      — количество дней с прошлого контакта
previous   — количество предыдущих контактов
poutcome   — результат предыдущей кампании
```

---

## Почему был удалён признак `duration`

В датасете есть признак:

```text
duration — длительность последнего разговора с клиентом
```

На первый взгляд он может быть очень полезным: если клиент долго разговаривал с оператором, вероятность отклика может быть выше.

Однако в реальной задаче предварительного таргетинга этот признак использовать нельзя. Если банк хочет заранее понять, кому стоит звонить, он ещё не знает, сколько продлится будущий разговор.

Использование `duration` привело бы к **data leakage** — утечке информации из будущего. Модель получила бы признак, который доступен только после совершения контакта, и качество на тесте стало бы искусственно завышенным.

Поэтому `duration` был исключён из модели.

---

## Project pipeline

Проект построен как полный ML-пайплайн:

1. Exploratory Data Analysis;
2. анализ целевой переменной;
3. преобразование target `yes / no` в `1 / 0`;
4. удаление leakage-признака `duration`;
5. train/test split со stratification;
6. обработка признаков:
   - `StandardScaler` для числовых признаков;
   - `OneHotEncoder` для категориальных признаков;
7. обучение моделей:
   - Logistic Regression;
   - Random Forest;
   - Gradient Boosting;
8. сравнение моделей по ML-метрикам;
9. выбор финальной модели;
10. top-k targeting analysis;
11. интерпретация результатов с точки зрения банковской маркетинговой кампании.

---

## Preprocessing

В данных есть как числовые, так и категориальные признаки.  
Для корректной работы моделей использовался `ColumnTransformer`.

Числовые признаки масштабировались через:

```text
StandardScaler
```

Категориальные признаки кодировались через:

```text
OneHotEncoder(handle_unknown="ignore")
```

Такой подход позволяет объединить обработку разных типов признаков в одном пайплайне и избежать ручной подготовки признаков отдельно для train и test.

---

## Target distribution

Задача является несбалансированной: клиентов, которые оформили депозит, заметно меньше, чем клиентов без отклика.

Из-за этого accuracy не является основной метрикой. Если модель будет почти всегда предсказывать `no`, она может получить высокую accuracy, но будет бесполезна для поиска клиентов с высокой вероятностью отклика.

Поэтому в проекте используются следующие метрики:

```text
ROC-AUC
PR-AUC
precision
recall
F1-score
top-k response rate
lift
```

---

## Model comparison

Были обучены и сравнены три модели:

1. **Logistic Regression** — простая baseline-модель;
2. **Random Forest** — ансамбль деревьев решений;
3. **Gradient Boosting** — boosting-модель для табличных данных.

Результаты на test set:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.772 | 0.409 | 0.266 | 0.624 | 0.373 |
| Random Forest | 0.793 | 0.432 | 0.370 | 0.564 | 0.447 |
| Gradient Boosting | 0.799 | 0.451 | 0.670 | 0.190 | 0.296 |

Random Forest показал лучший F1-score, однако основная бизнес-задача проекта — не только бинарная классификация, а ранжирование клиентов по вероятности отклика.

Gradient Boosting показал лучшие значения ROC-AUC и PR-AUC, поэтому был выбран как финальная модель для top-k targeting.

---

## Почему важен top-k targeting

В маркетинговых задачах банк часто ограничен ресурсами:

- нельзя позвонить всем клиентам;
- операторы имеют ограниченное время;
- слишком частые коммуникации могут раздражать пользователей;
- важно повышать конверсию кампаний.

Поэтому модель используется не только для классификации, а для **приоритизации клиентов**.

Идея top-k targeting:

1. модель оценивает вероятность отклика для каждого клиента;
2. клиенты сортируются по вероятности отклика;
3. банк выбирает верхние 5%, 10%, 20% или 30%;
4. маркетинговая коммуникация направляется сначала наиболее перспективным клиентам.

---

## Top-k targeting results

Baseline response rate на тестовой выборке:

```text
11.7%
```

Это означает, что если выбирать клиентов случайно, средняя доля откликнувшихся будет около 11.7%.

Результаты top-k анализа:

| Top-k clients | Clients selected | Response rate | Baseline response rate | Lift | Positive responses |
|---:|---:|---:|---:|---:|---:|
| 5% | 452 | 0.619 | 0.117 | 5.295 | 280 |
| 10% | 904 | 0.512 | 0.117 | 4.378 | 463 |
| 20% | 1808 | 0.357 | 0.117 | 3.049 | 645 |
| 30% | 2712 | 0.276 | 0.117 | 2.357 | 748 |

---

## Business interpretation

Модель хорошо ранжирует клиентов по вероятности отклика.

Например, если выбрать top-10% клиентов по предсказанной вероятности, response rate составит:

```text
51.2%
```

При среднем отклике:

```text
11.7%
```

Это даёт lift:

```text
4.38
```

То есть top-10% клиентов, выбранных моделью, откликаются примерно в 4.38 раза чаще, чем случайно выбранные клиенты.

Такой подход может помочь банку:

- повысить конверсию маркетинговой кампании;
- снизить количество нерелевантных контактов;
- эффективнее использовать время операторов;
- сфокусироваться на клиентах с высокой вероятностью отклика.

---

## Results

Финальная модель — **Gradient Boosting**.

Основной результат проекта:

```text
top-10% клиентов по вероятности отклика дают response rate 51.2%
при baseline response rate 11.7%
и lift 4.38
```

Это показывает, что модель может быть полезна не только как классификатор, но и как инструмент приоритизации клиентов для банковской маркетинговой кампании.

---

## Project structure

```text
bank-marketing-response-prediction/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── 01_eda_modeling.ipynb
│
├── reports/
│   ├── README.md
│   ├── model_comparison.csv
│   ├── top_k_analysis.csv
│   └── figures/
│       ├── model_comparison_pr_auc.png
│       ├── top_k_response_rate.png
│       └── top_k_lift.png
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech stack

```text
Python
pandas
NumPy
scikit-learn
matplotlib
seaborn
joblib
Jupyter Notebook
Git
```

---

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset and place it into:

```text
data/raw/bank-full.csv
```

Run the notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_eda_modeling.ipynb
```

Alternatively, run the training script:

```bash
python src/train.py
```

Run prediction example:

```bash
python src/predict.py
```

---

## Generated artifacts

The project generates:

```text
reports/model_comparison.csv
reports/top_k_analysis.csv
reports/figures/model_comparison_pr_auc.png
reports/figures/top_k_response_rate.png
reports/figures/top_k_lift.png
models/gradient_boosting_response_model.joblib
```

Model files are not committed to the repository because they are reproducible from the training script.

---

## Future improvements

Possible next steps:

- add CatBoost / LightGBM comparison;
- add SHAP-based interpretation;
- add threshold tuning for binary classification;
- build a Streamlit demo for interactive client scoring;
- add feature importance analysis;
- add unit tests for preprocessing and top-k evaluation;
- save model metadata and experiment configuration.