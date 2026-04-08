import joblib
import pandas as pd

from utils import DATA_RAW, MODELS_DIR, OUTPUTS_DIR


def main():
    input_path = DATA_RAW / "claims_denial_synthetic.csv"
    model_path = MODELS_DIR / "denial_model.pkl"

    df = pd.read_csv(input_path)
    model = joblib.load(model_path)

    feature_cols = [
        "payer_type", "service_type", "provider_specialty", "place_of_service",
        "claim_amount", "patient_age", "submission_lag_days",
        "prior_auth_required", "prior_auth_obtained", "diagnosis_valid",
        "coding_accurate", "insurance_verified", "coverage_active",
        "duplicate_claim", "missing_modifier"
    ]

    score_df = df[["claim_id"] + feature_cols].copy()
    score_df["predicted_denial_probability"] = model.predict_proba(score_df[feature_cols])[:, 1]
    score_df["predicted_denial_flag"] = (score_df["predicted_denial_probability"] >= 0.50).astype(int)

    output_path = OUTPUTS_DIR / "scored_claims.csv"
    score_df.sort_values("predicted_denial_probability", ascending=False).to_csv(output_path, index=False)

    print(f"Scored claims saved to: {output_path}")
    print(score_df.sort_values('predicted_denial_probability', ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
