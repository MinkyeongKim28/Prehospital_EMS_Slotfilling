# EMS Call Slot-Filling

This repository provides a simplified pipeline for extracting structured prehospital variables from emergency medical service call transcripts using a large language model.

## Overview

The pipeline performs the following steps:

1. Load EMS call transcripts from a CSV or Excel file.
2. Extract predefined prehospital variables using GPT-4o.
3. Convert extracted information into standardized codes based on a predefined codebook.
4. Save reference labels and model predictions.
5. Evaluate slot-filling performance.

## Extracted Variables

The model extracts 21 prehospital variables, including:

* Age and gender
* Injury mechanism and injury type
* Intentionality
* Protective equipment use
* Occupation-related injury
* Initial AVPU scale
* Prehospital vital signs
* Accident location
* Route to hospital and mode of arrival
* Insurance type
* Prehospital cardiac arrest
* Prehospital notification status

The final evaluation uses 19 variables, excluding:

* Time of injury occurrence
* Prehospital notification status

## Project Structure

```text
.
├── data/
├── outputs/
│   ├── predictions/
│   └── evaluation/
├── src/
│   ├── run_slotfill.py
│   ├── slotfill_openai.py
│   ├── evaluate.py
│   ├── openai_config.py
│   └── slot_schema_ko.json
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
pandas
openpyxl
scikit-learn
openai
```

## OpenAI API Key

Set the OpenAI API key as an environment variable.

### Windows

```bash
set OPENAI_API_KEY=your_api_key
```

### Linux or macOS

```bash
export OPENAI_API_KEY=your_api_key
```

## Usage

```bash
python -m src.run_slotfill \
    --excel data/input.xlsx \
    --model gpt-4o \
    --codebook src/slot_schema_ko.json \
    --output outputs/predictions/predictions.csv
```

## Output Format

For each variable, the output contains:

```text
<variable>_truth
<variable>_pred
```

The LLM internally generates:

* `raw`: extracted text or normalized expression
* `codes`: standardized code used for evaluation

When a variable is not mentioned, it is assigned `unknown`. When it is mentioned but cannot be determined reliably, it is assigned `missing`.

## Evaluation

The evaluation script calculates:

* Exact-match accuracy
* Macro F1 score
* Weighted F1 score
* Unknown or no-value F1 score

For selected variables, `unknown` and `missing` are merged into a single `no_value` class during F1 evaluation.

## Notes

The code in this repository is a simplified version intended to demonstrate the overall workflow. The complete implementation includes additional preprocessing, variable-specific mapping rules, output validation, and exception handling.
