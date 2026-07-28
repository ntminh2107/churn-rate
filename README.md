# Customer Churn Demo

Project gồm notebook EDA/training, package ML tái sử dụng và một mini
application Streamlit để demo dự đoán churn.

## Chạy Streamlit app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Sau đó mở `http://localhost:8501`.

## Chức năng demo

- Sinh một khách hàng synthetic từ phân phối train bằng random seed.
- Cho phép chỉnh sửa toàn bộ model features trước khi dự đoán.
- Chọn model được highlight và điều chỉnh decision threshold.
- So sánh probability từ Tuned XGBoost, LightGBM và Random Forest.
- Hiển thị risk signals và raw customer record.
- EDA explorer qua dropdown: overview, churn distribution, numerical,
  categorical, correlation và rule audit.
- Hiển thị frozen best parameters và inference flow.

App không chạy lại hyperparameter search. Ba model được fit bằng best parameters
đã chốt trong notebook, sau đó được cache cho toàn bộ Streamlit session.

Model sử dụng 13 cột gốc và 15 engineered features đã chọn. Tám engineered
features có importance thấp được loại tập trung trong `PipelineConfig` để
training, tuning và Streamlit inference luôn dùng cùng một schema.

## Các entry point

- `streamlit_app.py`: demo application.
- `02_customer_churn_records_lightgbm.ipynb`: tuning/evaluation report.
- `pipeline/`: feature, model, tuning, inference và visualization modules.
- `python -m pipeline.run_pipeline`: chạy training workflow ngoài notebook.
