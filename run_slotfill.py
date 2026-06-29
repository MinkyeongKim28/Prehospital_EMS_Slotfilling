# src/run_slotfill.py

import json
import argparse
from pathlib import Path

import pandas as pd

from src.slotfill_openai import extract_slots_openai
from src.evaluate import (
    compute_metrics,
    compute_f1_metrics,
    save_mismatches,
)


# ============================================================
# Basic configuration
# ============================================================
TRANSCRIPT_COLUMN = "transcript_total"

TRUTH_ALIASES = {
    "Route to hospital_code": "Hospital visit route_code",
    "Prehospital cardiac arrest_code":
        "Pre-hospital cardiac arrest_code",
    "Prehospital notification status_code":
        "Pre-hospital notification_code",
    "Occupation-related_code":
        "Job-related_code",
    "Use of protective equipment_code":
        "Protective_code",
    "Prehospital Systolic blood pressure (mmHg)_value":
        "Sbp_value",
    "Prehospital Diastolic blood pressure (mmHg)_value":
        "Dbp_value",
    "Prehospital Pulse rate (beats per minute, bpm)_value":
        "Pr_value",
    "Prehospital Respiratory rate (breaths per minute, bpm)_value":
        "Rr_value",
    "Prehospital Body temperature (C)_value":
        "Bt_value",
    "Prehospital Oxygen saturation (%)_value":
        "Spo2_value",
}


# ============================================================
# Load input data
# ============================================================
def load_data(file_path, sheet_name=None):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    return pd.read_excel(
        file_path,
        sheet_name=sheet_name or 0,
    )


# ============================================================
# Load slot keys from the JSON schema
# ============================================================
def load_slot_keys(schema_path):
    with open(schema_path, encoding="utf-8") as file:
        schema = json.load(file)

    response_schema = schema[
        "json_schema_for_openai_responses_api"
    ]

    return list(
        response_schema["schema"]["properties"].keys()
    )


# ============================================================
# Prepare transcript data
# ============================================================
def prepare_transcript(df):
    df = df.copy()

    # Use transcript_1 when transcript_total is not available
    if TRANSCRIPT_COLUMN not in df.columns:
        df[TRANSCRIPT_COLUMN] = df["transcript_1"]

    # Replace missing values and remove extra spaces
    df[TRANSCRIPT_COLUMN] = (
        df[TRANSCRIPT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# Build the reference label DataFrame
# ============================================================
def build_truth(df, slot_keys):
    truth = {}

    for key in slot_keys:
        source_column = key

        # Use an alternative column name when needed
        if source_column not in df.columns:
            source_column = TRUTH_ALIASES.get(key)

        # Use unknown when the reference column is unavailable
        if source_column in df.columns:
            truth[f"{key}_truth"] = df[source_column]
        else:
            truth[f"{key}_truth"] = "unknown"

    return pd.DataFrame(truth)


# ============================================================
# Build the prediction DataFrame
# ============================================================
def build_predictions(predictions, slot_keys):
    prediction_rows = []

    for prediction in predictions:
        prediction = prediction or {}

        row = {
            f"{key}_pred": prediction.get(
                key,
                "unknown",
            )
            for key in slot_keys
        }

        prediction_rows.append(row)

    return pd.DataFrame(prediction_rows)


# ============================================================
# Main execution
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel",
        required=True,
        help="Path to the input CSV or XLSX file",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of rows to process",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--codebook",
        default="src/slot_schema_ko.json",
        help="Path to the slot schema",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions.csv",
        help="Path to the output CSV file",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. Load the input dataset
    # --------------------------------------------------------
    df = load_data(
        args.excel,
        args.sheet,
    )

    if args.n:
        df = df.head(args.n)

    df = prepare_transcript(df)

    # --------------------------------------------------------
    # 2. Load predefined slot names
    # --------------------------------------------------------
    slot_keys = load_slot_keys(args.codebook)

    # --------------------------------------------------------
    # 3. Perform LLM-based slot filling
    # --------------------------------------------------------
    predictions = extract_slots_openai(
        transcripts=df[TRANSCRIPT_COLUMN].tolist(),
        schema_path=args.codebook,
        model=args.model,
    )

    # --------------------------------------------------------
    # 4. Organize reference labels and predictions
    # --------------------------------------------------------
    df_truth = build_truth(
        df,
        slot_keys,
    )

    df_pred = build_predictions(
        predictions,
        slot_keys,
    )

    # --------------------------------------------------------
    # 5. Save prediction results
    # --------------------------------------------------------
    output = pd.concat(
        [
            df[[TRANSCRIPT_COLUMN]].reset_index(drop=True),
            df_truth.reset_index(drop=True),
            df_pred.reset_index(drop=True),
        ],
        axis=1,
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 6. Evaluate slot-filling performance
    # --------------------------------------------------------
    metrics = compute_metrics(
        df_pred,
        df_truth,
    )

    f1_metrics = compute_f1_metrics(
        df_pred,
        df_truth,
        drop_unknown=True,
    )

    save_mismatches(
        df_pred,
        df_truth,
        "outputs/mismatches.csv",
    )

    # Print summary results
    print("Overall metrics")
    print(metrics.get("_overall"))

    print("Overall F1")
    print(f1_metrics.get("_overall"))

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
