import numpy as np
import pandas as pd

from utils import DATA_RAW

RANDOM_SEED = 42
N_ROWS = 3000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_data(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    payer_type = rng.choice(["Medicare", "Medicaid", "Commercial", "Self-Pay"], size=n_rows, p=[0.22, 0.20, 0.50, 0.08])
    service_type = rng.choice(["Inpatient", "Outpatient", "Emergency", "Lab", "Radiology", "Surgery"], size=n_rows)
    provider_specialty = rng.choice(
        ["Primary Care", "Cardiology", "Oncology", "Orthopedics", "OB/GYN", "Pediatrics"],
        size=n_rows
    )
    place_of_service = rng.choice(["Hospital", "Clinic", "ASC", "Telehealth"], size=n_rows, p=[0.45, 0.35, 0.12, 0.08])

    claim_amount = rng.normal(1800, 950, n_rows).clip(80, 12000).round(2)
    patient_age = rng.integers(0, 91, size=n_rows)
    submission_lag_days = rng.integers(0, 46, size=n_rows)

    prior_auth_required = rng.choice([0, 1], size=n_rows, p=[0.58, 0.42])
    prior_auth_obtained = np.where(prior_auth_required == 1, rng.choice([0, 1], size=n_rows, p=[0.25, 0.75]), 1)
    diagnosis_valid = rng.choice([0, 1], size=n_rows, p=[0.08, 0.92])
    coding_accurate = rng.choice([0, 1], size=n_rows, p=[0.12, 0.88])
    insurance_verified = rng.choice([0, 1], size=n_rows, p=[0.10, 0.90])
    coverage_active = rng.choice([0, 1], size=n_rows, p=[0.07, 0.93])
    duplicate_claim = rng.choice([0, 1], size=n_rows, p=[0.94, 0.06])
    missing_modifier = rng.choice([0, 1], size=n_rows, p=[0.88, 0.12])

    linear_score = (
        1.8
        + 0.9 * (prior_auth_required == 1)
        - 1.4 * prior_auth_obtained
        - 1.8 * diagnosis_valid
        - 1.7 * coding_accurate
        - 1.2 * insurance_verified
        - 1.5 * coverage_active
        + 1.9 * duplicate_claim
        + 1.1 * missing_modifier
        + 0.03 * submission_lag_days
        + 0.00022 * claim_amount
        + 0.25 * (payer_type == "Medicaid")
        + 0.15 * (payer_type == "Medicare")
        + 0.20 * (service_type == "Surgery")
        + 0.18 * (service_type == "Emergency")
        + 0.12 * (place_of_service == "Hospital")
    )

    denial_probability = sigmoid(linear_score)
    denial_flag = rng.binomial(1, denial_probability, size=n_rows)

    df = pd.DataFrame({
        "claim_id": [f"CLM{100000 + i}" for i in range(n_rows)],
        "payer_type": payer_type,
        "service_type": service_type,
        "provider_specialty": provider_specialty,
        "place_of_service": place_of_service,
        "claim_amount": claim_amount,
        "patient_age": patient_age,
        "submission_lag_days": submission_lag_days,
        "prior_auth_required": prior_auth_required,
        "prior_auth_obtained": prior_auth_obtained,
        "diagnosis_valid": diagnosis_valid,
        "coding_accurate": coding_accurate,
        "insurance_verified": insurance_verified,
        "coverage_active": coverage_active,
        "duplicate_claim": duplicate_claim,
        "missing_modifier": missing_modifier,
        "denial_probability_simulated": denial_probability.round(4),
        "denial_flag": denial_flag
    })

    return df


if __name__ == "__main__":
    df = generate_data()
    output_path = DATA_RAW / "claims_denial_synthetic.csv"
    df.to_csv(output_path, index=False)
    print(f"Synthetic claims dataset saved to: {output_path}")
    print(df.head())
    print("\\nDenial rate:", round(df["denial_flag"].mean(), 3))
