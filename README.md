# Customer churn pipeline

Package này chứa toàn bộ logic được tách khỏi notebook
`02_customer_churn_records_lightgbm.ipynb`.

## Cấu trúc

```text
pipeline/
├── config.py                 # path, split, threshold và tuning config
├── contracts.py              # object truyền giữa các step
├── data.py                   # load, schema check, split 85/15
├── features.py               # ChurnFeatureEngineer và rule audit
├── preprocessing.py          # impute và one-hot encoder
├── model_lightgbm.py         # LightGBM baseline/search space
├── model_xgboost.py          # XGBoost baseline/search space
├── model_random_forest.py    # Random Forest baseline/search space
├── training.py               # orchestration của baseline models
├── tuning.py                 # train-only stratified CV
├── evaluation.py             # metrics, selection, external test
├── importance.py             # feature importance về source column
├── visualization.py          # toàn bộ report plots
├── reporting.py              # bảng và kết luận cuối
├── workflow.py               # API end-to-end không có plot
└── run_pipeline.py           # command-line entry point
```

## Chạy

Smoke test nhanh:

```bash
python -m pipeline.run_pipeline --quick --skip-baselines
```

Chạy đầy đủ cùng cấu hình notebook:

```bash
python -m pipeline.run_pipeline
```

Notebook vẫn là report chính, nhưng các cell chỉ điều phối các hàm ở trên.
Feature engineering và preprocessing tiếp tục được fit lại trong từng CV fold,
`CustomerId` và các cột định danh không đi vào model.
# churn-rate
