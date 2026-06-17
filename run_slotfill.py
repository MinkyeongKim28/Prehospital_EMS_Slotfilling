# src/run_slotfill.py
"""
run_slotfill.py (final, case-insensitive & robust mapping)
- pred: 코드북 21개 전부 예측(LLM/규칙기반)
- truth: 엑셀 헤더의 *_code 컬럼만 사용(정확히 일치 or 별칭 매칭). 없으면 unknown
- 헤더 정규화: 따옴표/전각/개행/여러 공백/nbsp/제로폭/오탈자('caradiac')/'_ code'→'_code'
- 매칭은 clean+lower 로 수행(대소문자/공백/철자요동에 강함)
- 열 순서: No, transcript_total → 21 truth(코드북 순서) → 21 pred(코드북 순서)
"""
import os, sys, json, argparse, re, unicodedata
import pandas as pd
from typing import Dict, List
from pathlib import Path

# --- robust imports (패키지/단독 실행 모두 지원) ---
try:
    from src.evaluate import compute_metrics, save_mismatches, compute_f1_metrics
    from src.slotfill_openai import extract_slots_openai
except ImportError:
    from .evaluate import compute_metrics, save_mismatches, compute_f1_metrics
    from .slotfill_openai import extract_slots_openai



# ----------------- 헤더/값 정규화 -----------------
def _clean_header(s: str) -> str:
    """엑셀/CSV 헤더 정규화:
    - 유니코드 NFKC
    - 제로폭/nbsp 제거
    - 따옴표/개행 제거, 연속 공백 1개로
    - 'caradiac' → 'cardiac' 교정
    - '_ code' → '_code' (예: 'Insurance type_ code')
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "").replace("\xa0", " ")
    s = s.replace('"', "").replace("'", "")
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("caradiac", "cardiac")
    s = s.replace("_ code", "_code")
    return s.strip()

def _clean_lower(s: str) -> str:
    return _clean_header(s).lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_header(c) for c in df.columns]
    return df


# ----------------- transcript_total 구성 -----------------
def ensure_transcript_total(df: pd.DataFrame, lang: str = "ko", first_script_only: bool = True) -> pd.DataFrame:  # ★ 기본 True
    out = df.copy()
    out.columns = [_clean_header(c) for c in out.columns]
    cols = list(out.columns)

    def _ll(x: str) -> str:
        return _clean_lower(x)

    base_idxs = [i for i, c in enumerate(cols) if _ll(c) == "transcript_total"]

    def _merge_from_base_indices(base_indexes):
        use_cols = []
        for idx in base_indexes:
            # 첫 번째 스크립트만: 기준 열만 사용
            if first_script_only:
                if cols[idx] not in use_cols:
                    use_cols.append(cols[idx])
                # 오른쪽 스캔 안 함
                continue

            # (기존 병합 로직: first_script_only=False일 때만 동작)
            if cols[idx] not in use_cols:
                use_cols.append(cols[idx])
            j = idx + 1
            while j < len(cols):
                nk = _ll(cols[j])
                if nk.endswith("_code") or nk.endswith("_value"):
                    break
                if nk.startswith("unnamed") or any(tok in nk for tok in ("transcript", "script", "stt")):
                    if cols[j] not in use_cols:
                        use_cols.append(cols[j])
                    j += 1
                    continue
                break

        if use_cols:
            s = out[use_cols].fillna("").astype(str).agg(" ".join, axis=1)
            out["transcript_total"] = s.str.replace(r"\s+", " ", regex=True).str.strip()
            return out, True
        return out, False

    if base_idxs:
        out, ok = _merge_from_base_indices(base_idxs)
        if ok:
            return out

    # ===== Fallback (transcript_* / script_* / stt_* 탐색) =====
    def collect(include_en: bool = False):
        cand = []
        for col in cols:
            nk = _ll(col)
            if nk.endswith("_code") or nk.endswith("_value"):
                continue
            if any(tok in nk for tok in ("transcript", "script", "stt")):
                if lang == "ko" and (not include_en):
                    if ("en" in nk) and ("ko" not in nk):
                        continue
                cand.append(col)

        parts, aux, totals = [], [], []
        for col in cand:
            nk = _ll(col)
            if re.search(r"transcript_[123]_(ko|en)$", nk):
                parts.append(col)
            elif re.search(r"(transcript_total|stt_total)_(ko|en)$", nk) or nk in ("transcript_total", "stt_total"):
                totals.append(col)
            else:
                aux.append(col)

        def part_index(c):
            m = re.search(r"_(\d)_", _ll(c))
            return int(m.group(1)) if m else 99

        parts = sorted(parts, key=part_index)
        return parts, aux, totals

    parts, aux, totals = collect(include_en=False)
    if not (parts or aux or totals):
        parts, aux, totals = collect(include_en=True)

    # 첫 번째만 선택하는 분기
    if first_script_only:
        if parts:
            use_cols = [parts[0]]
        elif totals:
            use_cols = [totals[0]]
        elif "transcript_total" in out.columns:
            use_cols = ["transcript_total"]
        elif aux:
            use_cols = [aux[0]]
        else:
            raise RuntimeError("transcript 관련 열을 찾지 못했습니다.")
    else:
        if parts or aux:
            use_cols = parts + [c for c in aux if c not in parts]
        elif totals:
            use_cols = totals
        elif "transcript_total" in out.columns:
            use_cols = ["transcript_total"]
        else:
            raise RuntimeError("transcript 관련 열을 찾지 못했습니다.")

    s = out[use_cols].fillna("").astype(str).agg(" ".join, axis=1)
    out["transcript_total"] = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return out






# ----------------- 코드북 21키 로드 -----------------
def load_codebook_keys(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    s = data.get("json_schema_for_openai_responses_api") or data
    props = (s.get("schema") or {}).get("properties") or {}
    return list(props.keys())  # 코드북 정의 순서 유지


# ----------------- 모델명 정규화 -----------------
def resolve_model(m: str) -> str:
    if not m:
        return "gpt-4o"
    name = m.strip().lower()
    aliases = {
        "4o": "gpt-4o", "gpt4o": "gpt-4o", "gpt-4o": "gpt-4o",
        "4o-mini": "gpt-4o-mini", "gpt-4o-mini": "gpt-4o-mini",
        "4.1": "gpt-4.1", "gpt-4.1": "gpt-4.1",
        "4.1-mini": "gpt-4.1-mini", "gpt-4.1-mini": "gpt-4.1-mini",
    }
    return aliases.get(name, m)


# ----------------- truth 매핑용: 코드북 정식 키 → 엑셀 헤더 별칭 -----------------
# (키/별칭 비교는 clean+lower 로 수행)
CODE_ONLY_ALIASES_LOWER: Dict[str, List[str]] = {
    # Route to hospital_code  ← Hospital visit route_code(엑셀)
    _clean_lower("Route to hospital_code"): [
        _clean_lower("Hospital visit route_code"),
        _clean_lower("Hospital visit route_code "),
        _clean_lower("내원 경로_code"),
        _clean_lower("내원경로_code"),
    ],

    # Prehospital cardiac arrest_code  ← Pre-hospital cardiac arrest_code / caradiac(오탈자)
    _clean_lower("Prehospital cardiac arrest_code"): [
        _clean_lower("Pre-hospital cardiac arrest_code"),
        _clean_lower("Pre-hospital caradiac arrest_code"),
        _clean_lower("병원전 심정지_code"),
        _clean_lower("병원 전 심정지_code"),
    ],

    # Prehospital notification status_code  ← Pre-hospital notification_code
    _clean_lower("Prehospital notification status_code"): [
        _clean_lower("Pre-hospital notification_code"),
        _clean_lower("Prehospital notification_code"),
        _clean_lower("병원전 사전통보 여부_code"),
        _clean_lower("사전통보_code"),
    ],

    # Occupation-related_code  ← Job-related_code
    _clean_lower("Occupation-related_code"): [
        _clean_lower("Job-related_code"),
        _clean_lower("업무 관련 여부_code"),
        _clean_lower("업무관련_code"),
    ],

    # Use of protective equipment_code  ← Protective_code
    _clean_lower("Use of protective equipment_code"): [
        _clean_lower("Protective_code"),
        _clean_lower("보호장비 사용_code"),
        _clean_lower("보호구착용_code"),
    ],

    # Initial AVPUa scale_code  ← (AVPU 축약 포함)
    _clean_lower("Initial AVPU scale_code"): [
        _clean_lower("Initial AVPU scale_code"),
        _clean_lower("Initial AVPUa scale_code"),
        _clean_lower("Initial AVPU scale_code"),
        _clean_lower("AVPU scale_code"),
        _clean_lower("Initial AVPU code"),
        _clean_lower("AVPU_code"),
        _clean_lower("avpu_code"),
        _clean_lower("의식수준_code"),
        _clean_lower("의식_code"),
    ],

    # Gender_code  ← gender_code
    _clean_lower("Gender_code"): [
        _clean_lower("gender_code"),
        _clean_lower("Gender_code"),
        _clean_lower("성별_code"),
    ],

    # Insurance type_code  ← Insurance type_code / Insurance type_ code
    _clean_lower("Insurance type_code"): [
        _clean_lower("Insurance type_code"),
        _clean_lower("Insurance type_ code"),
        _clean_lower("보험 유형_code"),
        _clean_lower("보험코드_code"),
    ],

    # ──────────────── (활력징후 _value 별칭 추가) ────────────────
    _clean_lower("Prehospital Systolic blood pressure (mmHg)_value"): [
        _clean_lower("Sbp_value"),
        _clean_lower("SBP_value"),
        _clean_lower("Systolic BP_value"),
        _clean_lower("systolic_value"),
        _clean_lower("수축기혈압_value"),
        _clean_lower("혈압수축기_value"),
        _clean_lower("수축기_value"),
    ],
    _clean_lower("Prehospital Diastolic blood pressure (mmHg)_value"): [
        _clean_lower("Dbp_value"),
        _clean_lower("DBP_value"),
        _clean_lower("Diastolic BP_value"),
        _clean_lower("diastolic_value"),
        _clean_lower("이완기혈압_value"),
        _clean_lower("혈압이완기_value"),
        _clean_lower("이완기_value"),
    ],
    _clean_lower("Prehospital Pulse rate (beats per minute, bpm)_value"): [
        _clean_lower("Pr_value"),
        _clean_lower("PR_value"),
        _clean_lower("Pulse_value"),
        _clean_lower("pulse_value"),
        _clean_lower("맥박_value"),
    ],
    _clean_lower("Prehospital Respiratory rate (breaths per minute, bpm)_value"): [
        _clean_lower("Rr_value"),
        _clean_lower("RR_value"),
        _clean_lower("Respiratory_value"),
        _clean_lower("resp_value"),
        _clean_lower("호흡_value"),
        _clean_lower("호흡수_value"),
    ],
    _clean_lower("Prehospital Body temperature (C)_value"): [
        _clean_lower("Bt_value"),
        _clean_lower("BT_value"),
        _clean_lower("Temp_value"),
        _clean_lower("temperature_value"),
        _clean_lower("체온_value"),
    ],
    _clean_lower("Prehospital Oxygen saturation (%)_value"): [
        _clean_lower("Spo2_value"),
        _clean_lower("SpO2_value"),
        _clean_lower("O2sat_value"),
        _clean_lower("산소포화도_value"),
    ],
    # 나머지(예: Age_code, Injury mechanism_code, Type of injury_code, Accident location_code, Mode of arrival_code)
    # 는 보통 엑셀과 동일 표기이므로 별칭 없음(정확 일치로 먼저 잡힘)

    # (추가) Time code - HHMM으로 저장
    _clean_lower("Time of injury occurrence_code"): [
    _clean_lower("Time_HHMM"),
    _clean_lower("Time HHMM"),
    _clean_lower("Injury time (HHMM)_value"),
    _clean_lower("Injury time"),
    _clean_lower("Accident time"),
    _clean_lower("발생시각"),
    _clean_lower("사고시각"),
    _clean_lower("발생시간"),
    _clean_lower("사고시간"),
    ]
}

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="CSV/XLSX 파일 경로")
    ap.add_argument("--sheet", default=None, help="엑셀 시트명(미지정 시 자동)")
    ap.add_argument("--n", type=int, default=878, help="상위 N개만 처리")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--codebook", default="src/slot_schema_ko.json")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out-csv", default="outputs/predictions/preds.csv")
    ap.add_argument("--out-metrics", default="outputs/evaluation/metrics.json")
    args = ap.parse_args()

    # 로그
    args.model = resolve_model(args.model)
    print("[DEBUG] argv =", sys.argv)
    print("[DEBUG] model =", args.model)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)

    # 데이터 로드
    if args.excel.lower().endswith(".csv"):
        df = pd.read_csv(args.excel)
        df = ensure_transcript_total(df, lang="ko", first_script_only=True)   # 첫 번째 script만 처리
        transcripts = df["transcript_total"].astype(str).tolist()
        used_sheet = None
    else:
        try:
            xl = pd.ExcelFile(args.excel)
        except Exception:
            xl = pd.ExcelFile(args.excel, engine="openpyxl")
        if args.sheet and args.sheet in xl.sheet_names:
            used_sheet = args.sheet
            df = xl.parse(args.sheet)
        else:
            prefer = [s for s in xl.sheet_names if any(k in s.lower() for k in ["total", "전체", "full", "all"])]
            used_sheet = prefer[0] if prefer else xl.sheet_names[0]
            df = xl.parse(used_sheet)
    print(f"[DATA] file='{args.excel}' sheet='{used_sheet}' rows={len(df)}")

    # 헤더 정규화 + transcript_total 생성
    df = normalize_columns(df)
    df = ensure_transcript_total(df, args.lang,first_script_only=True) 

    # 상위 N개 + No 보장
    sub = df.head(args.n).copy()
    if "No" not in sub.columns:
        sub.insert(0, "No", range(1, len(sub) + 1))

    transcripts = sub["transcript_total"].fillna("").astype(str).tolist()

    # NEW : metrics 계산용 정답 레코드 그대로 전달
    truth_rows = sub.to_dict("records")

    # 코드북 21키(순서 기준)
    all_21_keys = load_codebook_keys(args.codebook)

    # -------- pred: 21개 전부 예측 --------
    preds = extract_slots_openai(
        transcripts,
        schema_path=args.codebook,
        model=args.model,
        api_key=args.api_key,
        truth_rows=truth_rows,     # NEW : per-feature acc가 여기서 출력됩니다.
        write_preds=False,
        truth_aliases_lower=CODE_ONLY_ALIASES_LOWER 
    )


    rows = []
    for p in preds:
        row = {}
        p = p or {}
        for k, v in p.items():
            row[(k if k.endswith("_code") else k) + "_pred"] = v
        rows.append(row)
    df_pred = pd.DataFrame(rows)

    # pred 21개 강제 생성 + 정확히 21개만 남김
    pred_order  = [f"{k}_pred"  for k in all_21_keys]
    for c in pred_order:
        if c not in df_pred.columns:
            df_pred[c] = "unknown"
    df_pred = df_pred.fillna("unknown").replace({"": "unknown"})
    df_pred = df_pred[pred_order]

    # -------- truth: 파일의 *_code 컬럼만 사용, 나머지 unknown --------
    truth = sub.copy()

    # clean+lower → 원본 헤더
    col_map_lower = {_clean_lower(c): c for c in truth.columns}

    # --- (추가) unknown/missing을 '코드값'으로 쓰기 위한 스키마 룩업 준비 ---
    def _norm_text(s):  # 라벨 비교용
        return str(s).strip().lower()

    def _build_schema_lookup_for_truth(schema_path: str):
        """name_en -> {label_lower: code} 매핑(라벨→코드)"""
        data = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        lut = {}
        for v in data.get("variables", []):
            name = v.get("name_en")
            cats = v.get("categories", {})
            lab2code = {}
            for code, label in cats.items():
                lab2code[_norm_text(label)] = str(code)
            lut[name] = lab2code
        return lut

    _schema_lut = _build_schema_lookup_for_truth(args.codebook)

    def _unknown_code_for_truth(std_name_en: str) -> str:
        """각 변수(enum)에서 'unknown' 라벨의 코드값을 가져온다. 없으면 'unknown' 텍스트로 폴백"""
        lab2code = _schema_lut.get(std_name_en, {})
        return lab2code.get("unknown", "unknown")

    rename_map: Dict[str, str] = {}
    found_map: Dict[str, str] = {}
    missing_truth_keys: List[str] = []

    for key in all_21_keys:
        tcol = key + "_truth"
        k_lower = _clean_lower(key)

        # 1) 동일 키로 바로 존재하면 사용(정확 일치: clean+lower 기준)
        if k_lower in col_map_lower:
            src = col_map_lower[k_lower]
            rename_map[src] = tcol
            found_map[key] = src
            continue

        # 2) 별칭으로 시도(정확 일치: clean+lower 기준)
        matched = None
        for alias_lower in CODE_ONLY_ALIASES_LOWER.get(k_lower, []):
            if alias_lower in col_map_lower:
                matched = col_map_lower[alias_lower]
                break

        if matched:
            rename_map[matched] = tcol
            found_map[key] = matched
        else:
            truth[tcol] = "unknown"
            missing_truth_keys.append(key)

    truth = truth.rename(columns=rename_map)

    # 커버리지 로그
    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/truth_coverage.json","w",encoding="utf-8") as f:
        json.dump({
            "matched_from_excel": found_map,                    # codebook_key -> 실제 원본 헤더
            "missing_set_as_unknown": missing_truth_keys,       # 파일에 없어 unknown 처리된 키
            "excel_headers_cleaned_lower": list(col_map_lower.keys())[:200],
        }, f, ensure_ascii=False, indent=2)
    print("[COVERAGE] truth coverage saved: outputs/evaluation/truth_coverage.json")

    # -------- 저장/정렬 --------
    truth_order = [f"{k}_truth" for k in all_21_keys]
    for c in truth_order:
        if c not in truth.columns:
            truth[c] = "unknown"
    df_truth = truth[truth_order]

    # 앞쪽에 붙일 ID/메타 컬럼: (있으면) go, No, OCS등록번호 + transcript_total
    id_cols = []

    # 1) go가 있으면 제일 앞에
    if "go" in sub.columns:
        id_cols.append("go")

    # 2) 그 다음 No
    if "No" in sub.columns:
        id_cols.append("No")

    # 3) 그 다음 OCS등록번호 (있으면)
    if "OCS등록번호" in sub.columns:
        id_cols.append("OCS등록번호")

    # 4) 마지막으로 transcript_total
    id_cols.append("transcript_total")

    out = pd.concat(
        [
            sub[id_cols].reset_index(drop=True),
            df_truth.reset_index(drop=True),
            df_pred.reset_index(drop=True),
        ],
        axis=1,
    )

    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] 예측 저장: {args.out_csv}")

    # -------- 평가/불일치 --------
    metrics = compute_metrics(df_pred, df_truth)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] 메트릭 저장: {args.out_metrics}")
    print("[ACC] overall:", metrics.get("_overall", {}))

    mism_csv = "outputs/evaluation/mismatches_sample.csv"
    save_mismatches(df_pred, df_truth, mism_csv, id_series=sub["No"])
    print(f"[OK] 불일치 샘플 저장: {mism_csv}")

    # -------- F1 (labeled-only 기본) --------
    f1 = compute_f1_metrics(df_pred, df_truth, drop_unknown=True)
    with open("outputs/evaluation/metrics_f1.json", "w", encoding="utf-8") as f:
        json.dump(f1, f, ensure_ascii=False, indent=2)

    print("[F1] overall micro-f1(labeled-only):", f1.get("_overall", {}).get("micro_f1", 0.0))
    for k, v in f1.items():
        if k == "_overall": 
            continue
        print(f"[F1] {k:<55} micro={v['micro_f1']:.3f}  macro={v['macro_f1']:.3f}  weighted={v['weighted_f1']:.3f}  (n={v['n']})")



if __name__ == "__main__":
    main()
