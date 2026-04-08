import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import DATA_RAW, MODELS_DIR, OUTPUTS_DIR


def main():
    input_path = DATA_RAW / "claims_denial_synthetic.csv"
    df = pd.read_csv(input_path)

    target = "denial_flag"
    drop_cols = ["claim_id", "denial_probability_simulated", target]
    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_features = ["payer_type", "service_type", "provider_specialty", "place_of_service"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }

    model_path = MODELS_DIR / "denial_model.pkl"
    joblib.dump(pipeline, model_path)

    metrics_path = OUTPUTS_DIR / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    print("Model saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print("\\nKey Metrics")
    for key in ["accuracy", "precision", "recall", "roc_auc"]:
        print(f"{key}: {metrics[key]}")
    print("\\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
