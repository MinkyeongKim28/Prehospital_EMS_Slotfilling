"""
evaluate.py (robust matching)
- pred와 truth의 슬롯명을 정규화(공백/언더스코어/대소문자/동의어/오탈자)하여 1:1 매칭
- 'unknown'/'missing' 포함 비교 (동일하면 정답)
- 성별 정규화: 남=M, 여=W (F/female/여성 등은 W로 통일)
- Age_code: pred가 중간값(예: 16.5)일 때 truth가 floor/ceil(16 또는 17)이면 정답 처리
- 불일치 샘플 CSV 저장(save_mismatches)
"""

import re
import json
import math
from typing import Dict, Any, List, Tuple
import pandas as pd
from collections import Counter


# ── 한글/동의어/오탈자 매핑 테이블 ──
ALIASES = {
    "Age_code": ["나이코드", "연령코드", "연령_코드"],
    "gender_code": ["성별코드", "성별_코드", "gender"],
    "Intentionality_code": ["의도성코드", "의도코드", "의도성_코드"],
    "Injury mechanism_code": ["손상기전코드", "손상 기전 코드", "기전코드"],
    "Type of injury_code": ["손상유형코드", "손상 유형 코드", "유형코드"],
    "Initial AVPU scale_code": ["avpu코드", "의식수준코드", "의식코드"],
    "Accident location_code": ["사고장소코드", "장소코드"],
    "Hospital visit route_code": ["route to hospital_code", "병원방문경로코드", "내원경로코드"],
    "Mode of arrival_code": ["도착수단코드", "이송수단코드"],
    "Insurance type_code": ["보험유형코드", "보험코드"],
    "Pre-hospital cardiac arrest_code": ["pre-hospital caradiac arrest_code", "병원전심정지코드", "병원 전 심정지 코드"],
    "Pre-hospital notification_code": ["사전통보코드", "사전 통보 코드", "prehospital notification_code"],
    "Protective_code": ["보호구착용코드", "보호 장비 코드", "use of protective equipment_code"],
}

def _norm_token(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("caradiac", "cardiac")  # 오탈자 교정
    s = re.sub(r"[\W_]+", "", s)         # 공백/언더스코어/특수문자 제거
    return s

def get_col(df: pd.DataFrame, name: str) -> str:
    """엑셀에 섞여있는 code/라벨/한글 컬럼명을 최대한 찾아 매핑."""
    norm = lambda x: re.sub(r"[\s_]+", "", (x or "").lower())
    target = norm(name)
    # 1) 완전 일치
    for col in df.columns:
        if norm(col) == target:
            return col
    # 2) ALIAS 기반
    canonical = None
    for cano in ALIASES.keys():
        if _norm_token(name) == _norm_token(cano):
            canonical = cano
            break
    if canonical:
        tgt = _norm_token(canonical)
        for col in df.columns:
            if _norm_token(col) == tgt:
                return col
        for alias in ALIASES[canonical]:
            tgt2 = _norm_token(alias)
            for col in df.columns:
                if _norm_token(col) == tgt2:
                    return col
    # 3) 느슨한 포함
    target2 = _norm_token(name)
    for col in df.columns:
        cn = _norm_token(col)
        if target2 in cn or cn in target2:
            return col
    # 못 찾으면 원본명
    return name

def properties_from_codebook(codebook_path: str) -> List[str]:
    with open(codebook_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    s = data.get("json_schema_for_openai_responses_api") or data
    schema = s.get("schema", {}) or {}
    props = schema.get("properties", {}) or {}
    return list(props.keys())

def resolved_cols_from_codebook(df: pd.DataFrame, codebook_path: str) -> Dict[str, str]:
    """코드북의 21개 *_code 키 각각을 엑셀의 실제 컬럼명으로 매핑. 없으면 None."""
    props = properties_from_codebook(codebook_path)
    mapping = {}
    for key in props:
        col = get_col(df, key)
        mapping[key] = col if col in df.columns else None
    return mapping


def _base_name(col: str) -> str:
    """_pred/_truth/_code 접미사 제거 후 표준화 키 생성"""
    c = re.sub(r"_(pred|truth)$", "", (col or ""))
    c = re.sub(r"_code$", "", c)
    return _norm_token(c)

def _pair_fields(df_pred: pd.DataFrame, df_truth: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    pred의 *_pred 컬럼과 truth의 *_truth 컬럼을 표준화 키로 매칭하여 (pcol, tcol, canonical) 리스트 반환
    """
    pred_cols = [c for c in df_pred.columns if c.endswith("_pred")]
    truth_cols = [c for c in df_truth.columns if c.endswith("_truth")]

    pred_map = {_base_name(c): c for c in pred_cols}
    truth_map = {_base_name(c): c for c in truth_cols}

    common = sorted(set(pred_map.keys()) & set(truth_map.keys()))
    pairs = [(pred_map[k], truth_map[k], k) for k in common]
    return pairs

# ---------- 값 정규화 ----------
def _normalize_value(x):
    # NaN, None, 빈문자 → unknown
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return "unknown"

    s = str(x).strip()
    if s == "":
        return "unknown"

    low = s.lower()

    # 성별 표준화
    if low in ["m", "male", "남", "남성", "man", "boy"]:
        return "M"
    if low in ["f", "female", "여", "여성", "woman", "girl", "w"]:
        return "W"

    # 결측류 → unknown
    if low in ["na", "n/a", "none", "null", "-", "없음", "미상", "unknown"]:
        return "unknown"

    if low in ["missing", "miss"]:
        return "missing"

    return s


def _to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None

def _age_equal(a: str, b: str) -> bool:
    """
    Age_code 특수 비교:
    - pred가 16.5 처럼 중간값일 때 truth가 16 또는 17이면 정답
    - 양쪽이 정수 코드 문자열이면 동일성 비교
    """
    af = _to_float(a)
    bf = _to_float(b)
    if af is None or bf is None:
        # 정수/문자 조합일 수 있으므로 정규화 문자열 비교로도 체크
        return str(_normalize_value(a)) == str(_normalize_value(b))
    # pred가 .5면 floor/ceil 중 하나와 같으면 정답
    if abs(af - round(af)) == 0.5:
        from math import floor, ceil
        return bf in (floor(af), ceil(af))
    return af == bf

# ---------- 원본에서 truth 컬럼 해석 ----------
RESOLVED = None  # NOTE: 전역 캐시는 데이터프레임이 바뀌면 부정확해질 수 있습니다.

def resolved_cols(df: pd.DataFrame)->Dict[str,str]:
    """
    과거 하위호환용 매핑(일부 핵심 슬롯). 현재 파이프라인은 코드북 기반 resolved_cols_from_codebook 사용을 권장.
    """
    global RESOLVED
    if RESOLVED:
        return RESOLVED
    base_names = [
        "Age_code",
        "gender_code",
        "Intentionality_code",
        "Injury mechanism_code",
        "Type of injury_code",
        "Initial AVPU scale_code",
        "Accident location_code",
        "Hospital visit route_code",
        "Mode of arrival_code",
        "Insurance type_code",
        "Pre-hospital cardiac arrest_code",
        "Pre-hospital notification_code",
        "Protective_code",
    ]
    m = {}
    for b in base_names:
        col = get_col(df, b)
        m[b] = col
    RESOLVED = m
    return RESOLVED

# ---------- 활력징후 binning (truth) ----------
def _bin(value, bins):
    if value is None:
        return None
    for code, cond in bins:
        if cond(value):
            return code
    return None

def _num(x):
    try:
        return float(x)
    except Exception:
        return None

def vitals_truth(df: pd.DataFrame)->pd.DataFrame:
    sbp_bins=[("3",lambda v:v==0),("4",lambda v:1<=v<=49),("5",lambda v:50<=v<=75),("6",lambda v:76<=v<=89),("7",lambda v:v>=89)]
    dbp_bins=[("3",lambda v:v==0),("4",lambda v:1<=v<=29),("5",lambda v:30<=v<=45),("6",lambda v:46<=v<=59),("7",lambda v:v>=59)]
    pr_bins =[("3",lambda v:v==0),("4",lambda v:1<=v<=29),("5",lambda v:30<=v<=59),("6",lambda v:60<=v<=100),("7",lambda v:101<=v<=119),("8",lambda v:v>=119)]
    rr_bins =[("3",lambda v:v==0),("4",lambda v:1<=v<=5),("5",lambda v:6<=v<=9),("6",lambda v:10<=v<=29),("7",lambda v:v>=29)]
    bt_bins =[("3",lambda v:v==0),("4",lambda v:0<v<=24.0),("5",lambda v:24.1<=v<=28.0),("6",lambda v:28.1<=v<=32.0),("7",lambda v:32.1<=v<=35.0),("8",lambda v:35.1<=v<=37.8),("9",lambda v:v>37.8)]
    spo2_bins=[("3",lambda v:v==0),("4",lambda v:0<v<=80),("5",lambda v:81<=v<=90),("6",lambda v:91<=v<=95),("7",lambda v:v>95)]
    out = pd.DataFrame(index=df.index)
    for col, bins in [
        ("Sbp_value", sbp_bins), ("Dbp_value", dbp_bins), ("Pr_value", pr_bins),
        ("Rr_value", rr_bins), ("Bt_value", bt_bins), ("Spo2_value", spo2_bins)
    ]:
        if col in df.columns:
            out[col.replace("_value","_code_truth")] = df[col].map(lambda x: _bin(_num(x), bins))
    return out


# ---------- 메트릭/불일치 ----------
def _values_equal(slot_base: str, a, b) -> bool:
    a = _normalize_value(a)  # pred
    b = _normalize_value(b)  # truth

    # 변경 -> truth가 unknown(=none/빈칸/NaN 등 포함)이고 pred가 unknown이면 정답
    if b == "unknown" and a == "unknown":
        return True

    # Age 특수 로직
    if _base_name(slot_base) == _base_name("Age_code"):
        return _age_equal(a, b)

    return a == b


def compute_metrics(df_pred: pd.DataFrame, df_truth: pd.DataFrame)->Dict[str,Any]:
    pairs = _pair_fields(df_pred, df_truth)
    summary={}
    total=0; correct=0
    for pcol, tcol, base in pairs:
        v = df_pred[[pcol]].join(df_truth[[tcol]])
        if len(v)==0:
            summary[base]={"acc": 0.0, "n": 0}
            continue
        match = v.apply(lambda r: _values_equal(base, r[pcol], r[tcol]), axis=1)
        acc = match.mean()
        summary[base]={"acc": float(acc), "n": int(len(v))}
        total += len(v); correct += int(match.sum())
    summary["_overall"]={"acc": float(correct/total) if total else 0.0, "n": int(total)}
    return summary

def save_mismatches(df_pred: pd.DataFrame, df_truth: pd.DataFrame, out_csv: str, id_series=None, max_rows=200):
    pairs = _pair_fields(df_pred, df_truth)
    rows=[]
    for pcol, tcol, base in pairs:
        v = df_pred[[pcol]].join(df_truth[[tcol]])
        if len(v)==0: 
            continue
        mask = ~v.apply(lambda r: _values_equal(base, r[pcol], r[tcol]), axis=1)
        bad = v[mask].copy()
        if bad.empty:
            continue
        bad["slot"] = base
        bad["pred_norm"] = bad[pcol].map(_normalize_value)
        bad["truth_norm"] = bad[tcol].map(_normalize_value)
        if id_series is not None:
            bad["No"] = id_series
        cols = ["slot", pcol, tcol, "pred_norm", "truth_norm"]
        if "No" in bad.columns:
            cols = ["No"] + cols
        rows.append(bad[cols])
    # 결과 저장 (없어도 빈 CSV 만들어 주기)
    if rows:
        out = pd.concat(rows).head(max_rows).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["No","slot","pred","truth","pred_norm","truth_norm"])
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


# ----------------- F1-score 출력 ----------------------
def _labels_for_f1(series_pred, series_truth, drop_unknown=True):
    """라벨 목록(y_pred, y_true) 생성. drop_unknown=True면 unknown/missing 제거."""
    def _norm(v):
        s = _normalize_value(v)  # 기존 정규화 사용
        if drop_unknown and s in {"unknown", "missing"}:
            return None
        return s
    y_pred = []
    y_true = []
    for p, t in zip(series_pred, series_truth):
        p2, t2 = _norm(p), _norm(t)
        if p2 is None or t2 is None:
            continue
        y_pred.append(p2)
        y_true.append(t2)
    return y_pred, y_true

def _prf_from_counts(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def _multiclass_prf(y_pred, y_true, labels=None):
    """macro / weighted / micro PRF (멀티클래스)"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    support = Counter(y_true)

    # per-class
    per = {}
    for c in labels:
        tp = sum(1 for p,t in zip(y_pred,y_true) if p==c and t==c)
        fp = sum(1 for p,t in zip(y_pred,y_true) if p==c and t!=c)
        fn = sum(1 for p,t in zip(y_pred,y_true) if p!=c and t==c)
        prec, rec, f1 = _prf_from_counts(tp, fp, fn)
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support[c]}

    # macro / weighted
    if labels:
        macro_f1 = sum(per[c]["f1"] for c in labels) / len(labels)
        total = sum(per[c]["support"] for c in labels) or 1
        weighted_f1 = sum(per[c]["f1"]*per[c]["support"] for c in labels) / total
    else:
        macro_f1 = weighted_f1 = 0.0

    # micro
    tp = sum(sum(1 for p,t in zip(y_pred,y_true) if p==c and t==c) for c in labels)
    fp = sum(sum(1 for p,t in zip(y_pred,y_true) if p==c and t!=c) for c in labels)
    fn = sum(sum(1 for p,t in zip(y_pred,y_true) if p!=c and t==c) for c in labels)
    micro_prec, micro_rec, micro_f1 = _prf_from_counts(tp, fp, fn)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "per_class": per,
        "labels": labels,
    }

def compute_f1_metrics(df_pred: pd.DataFrame, df_truth: pd.DataFrame, drop_unknown=True)->Dict[str,Any]:
    """
    기존 compute_metrics(정확도)와 동일 페어링 규칙을 사용해 F1을 계산.
    - drop_unknown=True: truth/pred가 unknown/missing인 행은 제외(labeled-only)
    - False로 주면 unknown/missing도 하나의 클래스 취급
    """
    pairs = _pair_fields(df_pred, df_truth)
    out = {}
    micro_tp = micro_fp = micro_fn = 0  # overall micro 용
    overall_labels = set()
    overall_support = 0

    for pcol, tcol, base in pairs:
        y_pred, y_true = _labels_for_f1(df_pred[pcol], df_truth[tcol], drop_unknown=drop_unknown)
        if not y_true:
            out[base] = {"micro_f1": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "n": 0}
            continue
        res = _multiclass_prf(y_pred, y_true)
        out[base] = {
            "micro_f1": res["micro_f1"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "n": len(y_true),
        }
        # overall micro 집계
        labels = res["labels"]
        overall_labels.update(labels)
        # class별 tp/fp/fn 합산
        for c in labels:
            tp = sum(1 for p,t in zip(y_pred,y_true) if p==c and t==c)
            fp = sum(1 for p,t in zip(y_pred,y_true) if p==c and t!=c)
            fn = sum(1 for p,t in zip(y_pred,y_true) if p!=c and t==c)
            micro_tp += tp; micro_fp += fp; micro_fn += fn
        overall_support += len(y_true)

    # overall micro-F1
    micro_prec, micro_rec, micro_f1 = _prf_from_counts(micro_tp, micro_fp, micro_fn)
    out["_overall"] = {"micro_f1": micro_f1, "n": overall_support}
    return out