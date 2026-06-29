import pandas as pd
from sklearn.metrics import f1_score


# ============================================================
# Variable-specific prediction code remapping
# ============================================================
PRED_CODE_REMAP = {
    "Intentionality_code": {
        "6": "missing",
    },
    "Injury mechanism_code": {
        "99": "unknown",
        "16": "missing",
    },
    "Type of injury_code": {
        "9": "unknown",
        "6": "missing",
    },
    "Use of protective equipment_code": {
        "9": "unknown",
    },
    "Occupation-related_code": {
        "9": "unknown",
        "4": "missing",
    },
    "Initial AVPUa scale_code": {
        "n": "unknown",
    },
    "Accident location_code": {
        "99": "unknown",
        "22": "missing",
    },
    "Route to hospital_code": {
        "9": "unknown",
        "6": "missing",
    },
    "Mode of arrival_code": {
        "9": "unknown",
        "10": "missing",
    },
    "Insurance type_code": {
        "99": "unknown",
        "100": "missing",
    },
    "Prehospital cardiac arrest_code": {
        "9": "unknown",
        "5": "missing",
    },
}


# ============================================================
# Variables in which unknown and missing are treated as one class
# ============================================================
MERGED_NO_VALUE_VARS = {
    "Age_code",
    "Gender_code",
    "Prehospital Systolic blood pressure (mmHg)_value",
    "Prehospital Diastolic blood pressure (mmHg)_value",
    "Prehospital Pulse rate (beats per minute, bpm)_value",
    "Prehospital Respiratory rate (breaths per minute, bpm)_value",
    "Prehospital Body temperature (C)_value",
    "Prehospital Oxygen saturation (%)_value",
}


# ============================================================
# Value normalization
# ============================================================
def normalize_basic(value):
    """Normalize missing expressions and numeric strings."""

    if pd.isna(value):
        return "unknown"

    value = str(value).strip().lower()

    unknown_tokens = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "unknown",
        "unk",
        "not mentioned",
        "not_mentioned",
        "not mentioned in transcript",
        "no mention",
    }

    missing_tokens = {
        "missing",
        "miss",
        "unclear",
        "ambiguous",
        "not clear",
        "cannot determine",
        "not determined",
    }

    if value in unknown_tokens:
        return "unknown"

    if value in missing_tokens:
        return "missing"

    # Convert numeric strings such as 99.0 to 99
    try:
        numeric_value = float(value)

        if numeric_value.is_integer():
            return str(int(numeric_value))

        return str(numeric_value)

    except ValueError:
        return value


def normalize_truth(value, var_name=None):
    """Normalize a reference value."""

    return normalize_basic(value)


def normalize_pred(value, var_name=None):
    """Normalize a prediction and apply variable-specific remapping."""

    value = normalize_basic(value)

    if var_name in PRED_CODE_REMAP:
        value = PRED_CODE_REMAP[var_name].get(
            value,
            value,
        )

    return value


# ============================================================
# Exact-match accuracy
# ============================================================
def calculate_exact_match_accuracy(
    y_true,
    y_pred,
    var_name,
):
    """
    Calculate exact-match accuracy.

    For variables in MERGED_NO_VALUE_VARS:
        prediction unknown or missing matches truth unknown.

    For the other variables:
        only prediction unknown matches truth unknown.

    Non-no-value predictions must exactly match the truth.
    """

    y_true = pd.Series(y_true).apply(
        lambda value: normalize_truth(
            value,
            var_name,
        )
    )

    y_pred = pd.Series(y_pred).apply(
        lambda value: normalize_pred(
            value,
            var_name,
        )
    )

    if var_name in MERGED_NO_VALUE_VARS:
        pred_no_value_tokens = {
            "unknown",
            "missing",
        }
    else:
        pred_no_value_tokens = {
            "unknown",
        }

    matches = []

    for truth_value, pred_value in zip(
        y_true,
        y_pred,
    ):
        pred_is_no_value = (
            pred_value in pred_no_value_tokens
        )

        truth_is_no_value = (
            truth_value == "unknown"
        )

        if pred_is_no_value and truth_is_no_value:
            matches.append(True)

        elif (
            not pred_is_no_value
            and truth_value == pred_value
        ):
            matches.append(True)

        else:
            matches.append(False)

    return {
        "accuracy": sum(matches) / len(matches)
        if matches else 0.0,
        "n": len(matches),
        "n_correct": int(sum(matches)),
    }


# ============================================================
# F1 class conversion
# ============================================================
def convert_f1_class(
    value,
    var_name,
):
    """
    Merge unknown and missing into no_value for selected variables.
    """

    if (
        var_name in MERGED_NO_VALUE_VARS
        and value in {"unknown", "missing"}
    ):
        return "no_value"

    return value


def calculate_class_f1(
    y_true,
    y_pred,
    target_class,
):
    """Calculate one-vs-rest F1 for a target class."""

    y_true_binary = [
        int(value == target_class)
        for value in y_true
    ]

    y_pred_binary = [
        int(value == target_class)
        for value in y_pred
    ]

    return f1_score(
        y_true_binary,
        y_pred_binary,
        average="binary",
        zero_division=0,
    )


# ============================================================
# Macro, weighted, and unknown/no-value F1
# ============================================================
def calculate_f1_metrics(
    y_true,
    y_pred,
    var_name,
):
    """
    Calculate:
    - Macro F1
    - Weighted F1
    - Unknown or no-value F1
    """

    y_true = pd.Series(y_true).apply(
        lambda value: normalize_truth(
            value,
            var_name,
        )
    )

    y_pred = pd.Series(y_pred).apply(
        lambda value: normalize_pred(
            value,
            var_name,
        )
    )

    y_true_f1 = y_true.apply(
        lambda value: convert_f1_class(
            value,
            var_name,
        )
    )

    y_pred_f1 = y_pred.apply(
        lambda value: convert_f1_class(
            value,
            var_name,
        )
    )

    labels = sorted(
        set(y_true_f1.unique())
        | set(y_pred_f1.unique())
    )

    macro_f1 = f1_score(
        y_true_f1,
        y_pred_f1,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true_f1,
        y_pred_f1,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    target_class = (
        "no_value"
        if var_name in MERGED_NO_VALUE_VARS
        else "unknown"
    )

    unknown_or_no_value_f1 = calculate_class_f1(
        y_true=y_true_f1,
        y_pred=y_pred_f1,
        target_class=target_class,
    )

    target_support = int(
        (y_true_f1 == target_class).sum()
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "unknown_or_no_value_f1":
            unknown_or_no_value_f1,
        "unknown_evaluation_class":
            target_class,
        "unknown_or_no_value_support":
            target_support,
        "n": len(y_true_f1),
        "n_classes": len(labels),
        "classes": labels,
    }
