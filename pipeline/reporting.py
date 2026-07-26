"""Build final tables and narrative without model logic in the notebook."""

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.contracts import (
    DataBundle,
    ExternalEvaluation,
    FeatureImportanceResult,
    SelectionResult,
)


BASELINE_NAME_BY_ALGORITHM = {
    "LightGBM": "Engineered LGBM",
    "XGBoost": "Engineered XGBoost",
    "Random Forest": "Engineered Random Forest",
}


def build_final_comparison(selection: SelectionResult) -> pd.DataFrame:
    columns = [
        "Model",
        "Algorithm",
        "Imbalance strategy",
        "CV PR-AUC mean",
        "CV PR-AUC std",
        "Search seconds",
        "Best iteration",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Balanced Accuracy",
        "ROC-AUC",
        "PR-AUC",
    ]
    comparison = selection.ranking[columns].copy()
    comparison.insert(
        0,
        "Selected",
        comparison["Model"].eq(selection.selected_model_name),
    )
    return (
        comparison.sort_values(
            ["Selected", "CV PR-AUC mean"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def build_conclusion_markdown(
    config: PipelineConfig,
    data: DataBundle,
    validation_results: pd.DataFrame,
    selection: SelectionResult,
    external: ExternalEvaluation,
    importance: FeatureImportanceResult,
) -> str:
    comparison_index = validation_results.set_index("Model")
    selected_validation = comparison_index.loc[
        selection.selected_model_name
    ]
    baseline_name = BASELINE_NAME_BY_ALGORITHM[
        selection.artifact.algorithm
    ]
    baseline_validation = comparison_index.loc[baseline_name]
    cv_leader = selection.ranking.iloc[0]
    selected_cv_gap = (
        cv_leader["CV PR-AUC mean"]
        - selected_validation["CV PR-AUC mean"]
    )
    test_row = external.results.iloc[0]
    top_features = importance.source.head(8)["Source feature"].tolist()

    return f"""
# 10. Kết luận tuning

- Protocol: **{len(data.X_train):,} train / {len(data.X_validation):,} validation** từ source data và **{len(data.X_test):,} external test**.
- Tuning: **{config.tuning_iterations} random configurations x {config.cv_splits} stratified folds/model**, scoring bằng PR-AUC; feature engineering và preprocessing được fit lại trong từng fold.
- `RowNumber`, `CustomerId`, `Surname`, `Complain` và `Exited` không xuất hiện trong feature matrix.
- CV leader là **{cv_leader['Model']}** với PR-AUC **{cv_leader['CV PR-AUC mean']:.4f}**. **{selection.selected_model_name}** thấp hơn **{selected_cv_gap:.4f}**, nằm trong tolerance **{config.cv_tie_tolerance:.3f}**, nên được chọn nhờ validation F1 tốt nhất trong nhóm tương đương.
- Model được chọn **{selection.selected_model_name}** có mean CV PR-AUC **{selected_validation['CV PR-AUC mean']:.4f} ± {selected_validation['CV PR-AUC std']:.4f}**.
- Validation tại threshold **{selection.operating_threshold:.2f}**: Accuracy **{selected_validation['Accuracy']:.4f}**, Precision **{selected_validation['Precision']:.4f}**, Recall **{selected_validation['Recall']:.4f}**, F1 **{selected_validation['F1']:.4f}**, PR-AUC **{selected_validation['PR-AUC']:.4f}**.
- So với **{baseline_name}** chưa tuning ở cùng threshold: Precision thay đổi **{selected_validation['Precision'] - baseline_validation['Precision']:+.4f}**, Recall **{selected_validation['Recall'] - baseline_validation['Recall']:+.4f}**, F1 **{selected_validation['F1'] - baseline_validation['F1']:+.4f}**.
- Threshold F0.5 tham chiếu từ validation: **{selection.validation_f05_threshold:.4f}**; operating threshold ưu tiên recall được cố định theo quyết định nghiệp vụ: **{selection.operating_threshold:.2f}**.
- External test: Accuracy **{test_row['Accuracy']:.4f}**, Precision **{test_row['Precision']:.4f}**, Recall **{test_row['Recall']:.4f}**, F1 **{test_row['F1']:.4f}**, ROC-AUC **{test_row['ROC-AUC']:.4f}**, PR-AUC **{test_row['PR-AUC']:.4f}**.
- Top source features: **{', '.join(top_features)}**.

## Bảng tổng kết 3 tuned models trên validation

Các metric phụ thuộc quyết định dùng threshold **{selection.operating_threshold:.2f}**. Các model cách CV leader không quá **{config.cv_tie_tolerance:.3f}** được xem là tương đương và validation F1 được dùng để tie-break; external test không dùng để chọn model hoặc tham số.
"""
