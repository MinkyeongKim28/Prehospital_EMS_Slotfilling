# src/slotfill_openai.py

import json
from typing import Any, Dict, List

from openai import OpenAI

from src.openai_config import get_openai_client


# ============================================================
# Load slot names and allowed values from the codebook
# ============================================================
def load_codebook(
    codebook_path: str,
) -> tuple[List[str], Dict[str, List[str]]]:

    with open(codebook_path, encoding="utf-8") as file:
        codebook = json.load(file)

    response_schema = codebook[
        "json_schema_for_openai_responses_api"
    ]

    properties = response_schema["schema"]["properties"]

    slot_keys = list(properties.keys())

    slot_enums = {
        key: [
            str(value)
            for value in specification.get("enum", [])
        ]
        for key, specification in properties.items()
    }

    return slot_keys, slot_enums


# ============================================================
# Build a structured JSON output schema
# ============================================================
def build_output_schema(
    slot_keys: List[str],
    slot_enums: Dict[str, List[str]],
) -> Dict[str, Any]:

    code_properties = {}

    for key in slot_keys:
        allowed_values = slot_enums.get(key, [])

        if allowed_values:
            code_properties[key] = {
                "type": "string",
                "enum": list(
                    dict.fromkeys(
                        allowed_values
                        + ["unknown", "missing"]
                    )
                ),
            }
        else:
            code_properties[key] = {
                "type": "string",
            }

    raw_properties = {
        key: {"type": "string"}
        for key in slot_keys
    }

    return {
        "name": "slot_filling_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "raw": {
                    "type": "object",
                    "properties": raw_properties,
                    "required": slot_keys,
                    "additionalProperties": False,
                },
                "codes": {
                    "type": "object",
                    "properties": code_properties,
                    "required": slot_keys,
                    "additionalProperties": False,
                },
            },
            "required": ["raw", "codes"],
            "additionalProperties": False,
        },
    }


# ============================================================
# Build the slot-filling instruction
# Detailed variable rules are maintained in the codebook
# ============================================================
def build_prompt(
    slot_keys: List[str],
) -> str:

    slot_list = "\n".join(
        f"- {key}"
        for key in slot_keys
    )

    return f"""
You are an information extraction system for emergency
medical service call transcripts.

Extract all predefined prehospital variables from the
provided transcript.

Output two objects:

1. raw
   - The original expression or normalized text value.
   - Use "unknown" when the variable is not mentioned.
   - Use "missing" when it is mentioned but cannot be
     determined reliably.

2. codes
   - The final standardized code defined by the codebook.
   - Do not generate values outside the allowed schema.

Do not infer information that is not directly supported
by the transcript.

Required slots:
{slot_list}

Return only one JSON object containing "raw" and "codes".
""".strip()


# ============================================================
# Extract slots from one transcript
# ============================================================
def call_openai_slotfill(
    transcript: str,
    codebook_path: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> Dict[str, Dict[str, str]]:

    slot_keys, slot_enums = load_codebook(
        codebook_path
    )

    output_schema = build_output_schema(
        slot_keys,
        slot_enums,
    )

    prompt = build_prompt(slot_keys)

    client: OpenAI = get_openai_client(api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": transcript or "",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": output_schema,
        },
        temperature=0,
    )

    result = json.loads(
        response.choices[0].message.content
    )

    raw_output = result.get("raw", {})
    code_output = result.get("codes", {})

    # Ensure that all predefined slots are included
    raw_output = {
        key: str(raw_output.get(key, "unknown"))
        for key in slot_keys
    }

    code_output = {
        key: str(code_output.get(key, "unknown"))
        for key in slot_keys
    }

    return {
        "raw": raw_output,
        "final": code_output,
    }


# ============================================================
# Extract slots from multiple transcripts
# ============================================================
def extract_slots_openai(
    transcripts: List[str],
    schema_path: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
    **kwargs,
) -> List[Dict[str, str]]:

    predictions = []

    for index, transcript in enumerate(
        transcripts,
        start=1,
    ):
        result = call_openai_slotfill(
            transcript=transcript,
            codebook_path=schema_path,
            model=model,
            api_key=api_key,
        )

        predictions.append(result["final"])

        print(
            f"Processed transcript "
            f"{index}/{len(transcripts)}"
        )

    return predictions
