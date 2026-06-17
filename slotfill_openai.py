# src/slotfill_openai.py
import json, re
from typing import Dict, Any, List
from openai import OpenAI
import pandas as pd
import os
import time
import unicodedata

try:
    from src.openai_config import get_openai_client
except Exception:
    from .openai_config import get_openai_client

# Raw / Code 동시 출력 토글용
DUAL_OUTPUT = True

# --- 터미널 메트릭 출력 토글 ---
VERBOSE_METRICS = True 


# --- 호출별 토큰/시간 집계(배치 요약용) ---
METRICS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "elapsed_s": 0.0,
}

# =========================

# ■ 바이탈 베이스명(접미사 _code / _value 어떤 것이 와도 매칭)
# =========================
VITAL_BASENAMES = {
    "Prehospital Systolic blood pressure (mmHg)",
    "Prehospital Diastolic blood pressure (mmHg)",
    "Prehospital Pulse rate (beats per minute, bpm)",
    "Prehospital Respiratory rate (breaths per minute, bpm)",
    "Prehospital Body temperature (C)",
    "Prehospital Oxygen saturation (%)",
}
def _is_vital_prop(k: str) -> bool:
    return any(k.startswith(b) and k.endswith(("_code", "_value")) for b in VITAL_BASENAMES)

# 별칭 → 정식키
ALIAS_TO_CANON = {
    "gender": "gender_code",
    "initial avpu scale": "Initial AVPU scale_code",
    "initial avpu scale_code": "Initial AVPU scale_code",
    "initial avpua scale": "Initial AVPUa scale_code",
    "initial avpua scale_code": "Initial AVPUa scale_code",
    "avpu_code": "Initial AVPU scale_code",
    "injury mechanism_code": "Injury mechanism_code",
    "type of injury_code": "Type of injury_code",
    "accident location_code": "Accident location_code",
    "job-related_code": "Occupation-related_code",
    "occupation related_code": "Occupation-related_code",
}

# _truth 정규화
TRAILING_SUFFIXES = ("_truth", "_label", "_labels", "_gold", "_pred", "_prediction", "_gt")

def _get_truth_value_from_row(
    row: Dict[str, Any],
    key: str,
    KEYS_21: List[str],
    default: str = "unknown",
    truth_aliases_lower: Dict[str, List[str]] | None = None  # ★ 추가
):
    def _has_value(x):
        if x is None:
            return False
        try:
            if isinstance(x, float) and pd.isna(x):
                return False
        except Exception:
            pass
        return not (isinstance(x, str) and x.strip() == "")

    # 행의 컬럼 이름을 정규화 맵으로 만들어 둠
    name_norm_to_orig = {_norm_key(n): n for n in (row or {}).keys()}

    # 1) 기본 후보(자기 자신 + *_truth) + AVPU/AVPUa 교차
    cands = [key, f"{key}_truth"]
    if key in ("Initial AVPU scale_code", "Initial AVPUa scale_code"):
        other = "Initial AVPUa scale_code" if key == "Initial AVPU scale_code" else "Initial AVPU scale_code"
        cands += [other, f"{other}_truth"]

    # 1-a) 직접/정규화 일치 탐색
    for cand in cands:
        nn = _norm_key(cand)
        if nn in name_norm_to_orig:
            v = row.get(name_norm_to_orig[nn])
            if _has_value(v):
                return v

    # 2) 별칭 활용 탐색 (run_slotfill의 CODE_ONLY_ALIASES_LOWER를 주입받은 경우)
    if truth_aliases_lower:
        base = _norm_key(key)
        for alias in truth_aliases_lower.get(base, []):
            nn = _norm_key(alias)
            if nn in name_norm_to_orig:
                v = row.get(name_norm_to_orig[nn])
                if _has_value(v):
                    return v

    return default





def _norm_key(s: str) -> str:
    s = (s or "")
    # 추가: 흔한 CSV 깨짐 보정 (Â° → °), 중복 공백 제거
    s = s.replace("Â°", "°")
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in " _()%/,-").strip()



def _canon_key(k: str, KEYS_21: List[str]) -> str:
    # 1)s 평가용 접미사 제거
    k = _normalize_incoming_prop_key(k)

    # 2) 스키마에 있는 AVPU 키를 확인 (둘 중 하나)
    avpu_schema_key = "Initial AVPUa scale_code" if "Initial AVPUa scale_code" in KEYS_21 else "Initial AVPU scale_code"

    # 3) 들어온 키가 AVPU/AVPUa 변형이면, 스키마에 맞춰 강제 매핑
    nk = _norm_key(k)
    if _norm_key(avpu_schema_key) != nk:
        # 상대 변형 이름을 만들어 비교
        other_avpu = "Initial AVPU scale_code" if avpu_schema_key.endswith("AVPUa scale_code") else "Initial AVPUa scale_code"
        if nk == _norm_key(other_avpu):
            return avpu_schema_key  # 스키마에서 쓰는 쪽으로 맞춘다

    # 4) 그대로(접미사만 제거된) 키가 스키마에 있으면 반환
    if k in KEYS_21:
        return k

    # 5) 별칭 → 정식키
    for a, c in ALIAS_TO_CANON.items():
        if _norm_key(a) == nk:
            return c

    # 6) 느슨 일치
    for c in KEYS_21:
        if _norm_key(c) == nk:
            return c

    return None





def _normalize_incoming_prop_key(k: str) -> str:
    """
    외부에서 들어오는 키를 정리:
      - 평가용 접미사 제거: *_truth, *_label, ...
    """
    s = str(k or "")
    for suf in TRAILING_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s



# =========================
# 스키마/코드북 로딩
# =========================
def build_dual_schema(keys: List[str], enums: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    codes + raw 동시 출력용 스키마.
    - 바이탈: enum 없이 자유 문자열(원문/숫자 그대로 받기; 체온 보정은 모델이 codes에서만 수행)
    - 비바이탈: enum 강제 (unknown/missing 허용)
    - Age_code는 .5 중간값 허용
    """
    code_props = {}
    for k in keys:
        if _is_vital_prop(k):
            code_props[k] = {"type": "string"}
        else:
            allowed = sorted(set([str(v) for v in enums.get(k, [])] + ["unknown", "missing"]))
            if k.lower() == "age_code":
                half_codes = [str(x) for x in (4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
                                               12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
                                               18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5)]
                allowed = sorted(set(allowed).union(half_codes), key=lambda x: (len(x), x))
            code_props[k] = {"type": "string", "enum": allowed}

    raw_props = {k: {"type": "string"} for k in keys}

    return {
        "name": "slot_filling_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "object",
                    "properties": code_props,
                    "required": keys,
                    "additionalProperties": False,
                },
                "raw": {
                    "type": "object",
                    "properties": raw_props,
                    "required": keys,
                    "additionalProperties": False,
                },
            },
            "required": ["codes", "raw"],
            "additionalProperties": False,
        },
    }

def build_schema(keys: List[str], enums: Dict[str, List[str]]) -> Dict[str, Any]:
    props = {}
    for k in keys:
        if _is_vital_prop(k):
            props[k] = {"type": "string"}  # 바이탈은 자유 문자열
        else:
            allowed = sorted(set([str(v) for v in enums.get(k, [])] + ["unknown", "missing"]))
            if k.lower() == "age_code":
                half_codes = [str(x) for x in (4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
                                               12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
                                               18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5)]
                allowed = sorted(set(allowed).union(half_codes), key=lambda x: (len(x), x))
            props[k] = {"type": "string", "enum": allowed}

    return {
        "name": "slot_filling_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": props,
            "required": keys,
            "additionalProperties": False,
        },
    }

def load_codebook(path: str):
    """
    - KEYS_21: 스키마 키 순서 유지
    - enums:  스키마 enum 목록 그대로 (빈/none 제외)
    - cats:   4개 슬롯(Age/Mechanism/Type/AVPU)만 variables.categories 반영
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 스키마 키/enum
    s = data.get("json_schema_for_openai_responses_api") or data
    props = (s.get("schema") or {}).get("properties") or {}

    KEYS_21 = list(props.keys())
    enums: Dict[str, List[str]] = {}
    cats:  Dict[str, Dict[str, str]] = {}

    for k, spec in props.items():
        if not isinstance(spec, dict):
            continue
        e = spec.get("enum") or []
        enums[k] = [str(v) for v in e if str(v) not in {"", "none"}]

    # 2) variables.categories → cats (4개 슬롯만)
    variables = data.get("variables", [])
    for var in variables:
        name_en = var.get("name_en", "")
        categories = var.get("categories", {})
        if not name_en or not categories:
            continue

        if name_en == "Age":
            prop_key = "Age_code"
        elif name_en == "Injury mechanism":
            prop_key = "Injury mechanism_code"
        elif name_en == "Type of injury":
            prop_key = "Type of injury_code"
        elif name_en in ("Initial AVPU scale", "Initial AVPUa scale"):
            # 스키마에 실제로 들어있는 키를 선택
            if "Initial AVPUa scale_code" in KEYS_21:
                prop_key = "Initial AVPUa scale_code"
            else:
                prop_key = "Initial AVPU scale_code"

        elif name_en == "Accident location":
            prop_key = "Accident location_code"
        elif name_en in ("Occupation-related", "Job-related"):
            prop_key = "Occupation-related_code"

        else:
            continue  # 나머지는 프롬프트 규칙으로 처리

        if prop_key in KEYS_21:
            # 기본 categories를 code -> [label, ...] 구조로 변환
            labels = {}
            for code, v in (categories or {}).items():
                if isinstance(v, list):
                    labels[str(code)] = [str(x) for x in v]
                else:
                    labels[str(code)] = [str(v)]

            # label_aliases 병합 (code -> [alias...])
            aliases = var.get("label_aliases", {})
            for code, arr in (aliases or {}).items():
                labels.setdefault(str(code), [])
                if isinstance(arr, list):
                    labels[str(code)].extend(str(x) for x in arr)
                else:
                    labels[str(code)].append(str(arr))

            # 중복 제거(원본 순서 유지)
            for code, seq in labels.items():
                seen, dedup = set(), []
                for x in seq:
                    xx = str(x).strip()
                    if xx not in seen:
                        seen.add(xx)
                        dedup.append(xx)
                labels[code] = dedup

            cats[prop_key] = labels  # code -> [라벨/별칭들]




    return KEYS_21, enums, cats


# =========================
# 프롬프트 (기존 내용 최대 유지 + 체온 30대 보정 유지)
# =========================
def _prompt(keys: List[str], cats: Dict[str, Dict[str, str]]) -> str:
    vitals = [k for k in keys if _is_vital_prop(k)]
    key_lines = "- " + "\n- ".join(keys)

    # cats: 4개 슬롯만 들어 있음
    codebook_mappings = []
    for slot_name, mappings in cats.items():
        codebook_mappings.append(f"【{slot_name} 매핑】")
        for code, label in mappings.items():
            codebook_mappings.append(f"  {code}: {label}")
        codebook_mappings.append("\n")
    codebook_string = "\n".join(codebook_mappings)

    return (
        "당신은 응급의료 통화 대본에서 21개 슬롯을 채우는 정보추출기입니다.\n"
        "이 응답은 한 번만 출력하며, 반드시 두 블록을 모두 포함하세요:\n"
        "  • raw  : 각 슬롯의 라벨/원문 표현(바이탈은 숫자 문자열).\n"
        "  • codes: 각 슬롯의 최종 코드(enum). 스키마 enum과 정확히 일치해야 함(라벨 금지).\n"
        "세부 규칙의 해석/적용은 codes 블록에 한정합니다. 임의 추정 금지.\n\n"
        "  • unknown: 해당 슬롯 관련 **직접 언급이 없음**(데이터 없음).\n"
        "  • missing: 해당 슬롯 관련 **직접 언급은 있으나** '모름/확인불가/불명/충돌/환자 특정 불가/규칙상 판정 불가'로 값 확정 불가.\n"
        "  • (우선) 본 프롬프트 전체에서 unknown/missing은 위 정의를 최우선으로 따른다.\n"

        "【스코프/환자 고정】\n"
        "• 이 통화 스크립트 내용만 사용(추정·상식 보완 금지).\n"
        "• '환자/이분/남성/여성/아이' 등 **환자 지칭 + 상태/수치**의 **직접 진술**만 근거로 사용(전언/행정/타인 진술 제외).\n"

        "\n"

        "【핵심 원칙(모순 방지)】\n"
        "• raw는 사람이 이해하기 쉬운 표현, codes는 최종 코드(enum)입니다.\n"
        "• **codes는 raw와 논리적으로 모순되면 안 됩니다.**\n"
        "• 특히 아래 ‘결정형 슬롯’은 **codes를 임의 판단으로 선택 금지**: 반드시 스크립트 기반 규칙으로 산출하고, raw와 일치하도록 교정합니다.\n"
        f"  - 결정형 슬롯: Age_code, Initial AVPU scale_code, Prehospital cardiac arrest_code, 바이탈({', '.join(vitals)})\n"
        "• 비결정형 슬롯(그 외)은 codebook 요약을 근거로 codes를 선택하되, raw와 모순되면 raw를 수정하거나(근거 있을 때만) codes를 수정하세요.\n"
        "• 참고 가능한 codebook은 **아래 ‘요약 문자열’에 포함된 범위만**입니다(요약 밖 추정 금지).\n\n"

        "【raw 블록 작성 규칙】\n"
        f"• 아래 ‘필수 키 목록’을 그대로 사용해 21개 전 슬롯을 채우세요.\n"
        f"• 바이탈({', '.join(vitals)})은 **숫자만**(예: '120', '36.8')을 문자열로 기입합니다. 단위/문장 금지.\n"
        "• 비바이탈은 라벨/짧은 구절만. 숫자 코드 금지. **직접 언급 없음→'unknown'**, **직접 언급은 있으나 모름/불명확→'missing'**.\n\n"

        "【코드 변환 규칙(비결정형 슬롯용)】\n"
        "1) 각 슬롯마다 스크립트에서 해당 개념의 **라벨/표현**을 먼저 추출한다.\n"
        "2) 추출 라벨을 소문자/공백정리(예: '  교통 사고 '→'교통사고') 후, 아래 제공된 codebook 요약(categories/label_aliases)에 근거해 가장 잘 맞는 **코드값(enum)** 을 선택한다.\n"
        "   - 동의어·철자변형은 codebook 카테고리와 **의미가 동일한 항목**에만 매핑(없으면 추정 금지).\n"
        "3) **다중 후보**가 생기면: (i) 더 구체적인 항목(세분 코드) > (ii) 사건의 직접 원인(특히 Injury mechanism, Type) > (iii) 최신/핵심 단서 순으로 1개만 선택.\n"
        "4) codebook 요약 범위에서 **정확/유의미 매칭이 없으면**: **직접 언급 없음→'unknown'**, **직접 언급은 있으나 단일 코드 확정/매핑 불가→'missing'**. 임의 근사치 금지.\n"
        "5) 출력은 항상 **enum의 코드값(숫자/문자열)** 만. 라벨 원문은 출력하지 않는다.\n\n"

        "【Prehospital cardiac arrest_code — 결정형(과거/현재 포함 + unknown/missing)】\n"
        "• 출력 규칙:\n"
        "  - '1': 환자에게 심정지가 **과거/현재 포함하여 한 번이라도 발생**했다는 직접 진술이 있음(심정지, CPR, 제세동, 무맥/무호흡, asystole/PEA, ROSC/맥박·호흡 돌아옴 포함).\n"
        "  - '2': 심정지가 **오지 않았다/아니다/CPR·제세동 한 적 없다** 등 '심정지 이력 없음'의 직접 부정 진술이 있음.\n"
        "  - 'unknown': 심정지 관련 **직접 언급 자체가 없음**.\n"
        "  - 'missing': 심정지 관련 **직접 언급은 있으나** '모름/확인불가/의심/가능성/질문/가정/타인/대상 불명/진술 충돌'로 단일 확정 불가.\n"
        "• 단순 '의식 없음/무반응(unresponsive)'만으로는 심정지 발생으로 인정하지 않는다.\n"


        "────────────────────────────────────────────────────\n"

        "【필수 키 목록 — 정확히 이 이름으로만 출력】\n"
        f"{key_lines}\n\n"

        "【코드북 catefories, label_aliases 요약】\n"
        f"{codebook_string}\n"

        "────────────────────────────────────────────────────\n"

        "【Age_code — 결정형(‘년생’ 포함, codes 임의선택 금지)】\n"
        "■ 목표: **raw.Age_code(표준화된 나이표현)** 와 **codes.Age_code(코드값)** 이 규칙으로 1:1 대응하도록 만든다.\n"
        "■ 기본 원칙: 나이 표현이 불분명/충돌/환자 특정 불가이면 **raw.Age_code와 codes.Age_code 모두 'missing'**. "
        " 나이에 대한 언급 자체가 없을 경우 **raw.Age_code와 codes.Age_code 모두 'unknown'**.\n"
        "■ (중요) 나이 raw 추출은 반드시 아래 ‘선택 알고리즘’으로만 결정한다(임의 선택/상식 보완 금지).\n"
        "\n"
        "【0) 나이 표현 후보 수집(스크립트 전체에서 전부 찾기)】\n"
        "• 스크립트에서 아래 3종류 표현을 **모두** 수집한다(중간에 다른 값이 있어도 누락 금지).\n"
        "  - (T1) 년생: 'YYYY년생' 또는 'YY년생'\n"
        "  - (T2) 세/살: '만 N세', 'N세', 'N살' (정수 N)\n"
        "  - (T3) XX대: '30대', '30대 초반/중반/후반', 'X~Y대'(인접범위만)\n"
        "• 후보는 발견된 **텍스트 순서(등장 위치)** 를 유지한다.\n"
        "\n"
        "【1) raw.Age_code 선택 알고리즘(우선순위 + 하위타입 무시 + 마지막값)】\n"
        "① (타입 우선순위) 후보 중 (T1 년생)이 1개라도 있으면 **T1만 사용**한다.\n"
        "   - 이 경우 T2/T3가 더 나중에 등장해도 **절대 사용하지 않는다**(하위 타입 무시).\n"
        "② (다음 우선순위) T1이 없고 (T2 세/살)이 1개라도 있으면 **T2만 사용**한다.\n"
        "   - 이 경우 T3가 더 나중에 등장해도 **절대 사용하지 않는다**(하위 타입 무시).\n"
        "③ (마지막 우선순위) T1/T2가 모두 없을 때만 **T3(XX대)만 사용**한다.\n"
        "④ (마지막값 규칙) 선택된 타입(T1 또는 T2 또는 T3) 내부에서 후보가 2개 이상이면,\n"
        "   스크립트에서 **가장 마지막(가장 뒤)에 등장한 1개**만 raw.Age_code 결정에 사용한다.\n"
        "\n"
        "【2) raw.Age_code 표준화 규칙(선택된 ‘마지막 후보 1개’만 사용)】\n"
        "A. (T1: ‘…년생’이 선택된 경우)\n"
        "  - 선택된 마지막 후보에서 출생연도(YYYY 또는 YY)만 추출한다.\n"
        "  - YY 해석: 00–25 ⇒ 2000–2025, 26–99 ⇒ 1926–1999.\n"
        "  - 한국나이 N = (2025 - 출생연도) + 1.\n"
        "  - 1 ≤ N ≤ 130이면 raw.Age_code='N세', 범위 밖이면 raw.Age_code='missing'.\n"
        "\n"
        "B. (T2: ‘N세/살/만 N세’가 선택된 경우)\n"
        "  - 선택된 마지막 후보에서 정수 N만 추출한다(소수/분수 불허).\n"
        "  - 1 ≤ N ≤ 130이면 raw.Age_code='N세', 범위 밖이면 raw.Age_code='missing'.\n"
        "\n"
        "C. (T3: ‘XX대’가 선택된 경우)\n"
        "  - 수식어(초/중/후반)는 무시하고 'XX대'만 남긴다(예: '30대 초반'→'30대').\n"
        "  - 'X~Y대'는 **Y=X+10 인접 범위만** 허용하며 raw.Age_code='X대'.\n"
        "  - 허용 XX: 10,20,…,110(10의 배수). 그 외는 raw.Age_code='missing'.\n"
        "\n"
        "【3) codes.Age_code 산출(표 고정; raw에서만 계산)】\n"
        "• raw.Age_code가 'N세'일 때(정수 N):\n"
        "  <1:'1' / 1–4:'2' / 5–9:'3' / 10–14:'4' / 15–19:'5' / 20–24:'6' / 25–29:'7' /\n"
        "  30–34:'8' / 35–39:'9' / 40–44:'10' / 45–49:'11' / 50–54:'12' / 55–59:'13' /\n"
        "  60–64:'14' / 65–69:'15' / 70–74:'16' / 75–79:'17' / 80–84:'18' / 85–89:'19' /\n"
        "  90–94:'20' / 95–99:'21' / 100–104:'22' / 105–109:'23' / 110–114:'24' / 115–119:'25' / ≥120:'26'.\n"
        "• raw.Age_code가 'XX대'일 때:\n"
        "  10대=4.5, 20대=6.5, 30대=8.5, 40대=10.5, 50대=12.5, 60대=14.5, 70대=16.5, 80대=18.5, 90대=20.5, 100대=22.5, 110대=24.5.\n"
        "• raw.Age_code가 'unknown'이면 codes.Age_code도 'unknown'. raw.Age_code가 'missing'이면 codes.Age_code도 'missing'.\n"
        "\n"
        "【4) Age 자기검증(출력 직전 필수; 위반 시 즉시 unknown)】\n"
        "• (검증1) 선택된 타입의 근거(T1/T2/T3)가 스크립트에 실제로 존재하는가?\n"
        "  - 나이 관련 표현(T1/T2/T3) **자체가 전혀 없으면** raw/codes='unknown'.\n"
        "  - 나이 관련 표현은 **있지만** 선택 근거가 실제로 없거나 불일치하면 raw/codes='missing'.\n"
        "• (검증2) 상위 타입이 존재하는데 하위 타입으로 raw를 만들었는가?\n"
        "  - 나이 관련 표현이 아예 없으면 raw/codes='unknown'.\n"
        "  - 나이 관련 표현은 있으나 상위 타입을 무시한 경우 raw/codes='missing'.\n"
        "• (검증3) 선택된 타입 내부에서 ‘마지막 후보 1개’만 사용했는가?\n"
        "  - 나이 관련 표현이 아예 없으면 raw/codes='unknown'.\n"
        "  - 나이 관련 표현은 있으나 마지막 후보 1개 규칙을 위반하면 raw/codes='missing'.\n"
        "• (검증4) ‘스크립트 기반 계산 codes’와 ‘raw에서 표로 재계산한 codes’가 같은가?\n"
        "  - 나이 관련 표현이 아예 없으면 raw/codes='unknown'.\n"
        "  - 나이 관련 표현은 있으나 계산 결과가 다르면 raw/codes='missing'.\n"
        "• (검증5) raw.Age_code 형식이 'N세' 또는 'XX대'가 아니면:\n"
        "  - 나이 관련 표현이 아예 없으면 raw/codes='unknown'.\n"
        "  - 나이 관련 표현은 있으나 형식이 규칙을 위반하면 raw/codes='missing'.\n\n"


        "【AVPU(a) — 결정형(초엄격 직접진술 원칙)】\n"
        "• 판단 시점은 **통화 당시 현재 환자 1명**. **의식/멘탈에 대한 현재형 직접 표현**이 있을 때만 등급 판정.\n"
        "• **행동·생리 묘사만**(예: '말한다/대답한다/움직인다/눈 뜬다/숨 쉰다/맥박 있다/동공 반응/바이탈 정상')이면 → **반드시 'unknown'**.\n"
        "• **LOC/의식소실/실신** 등 과거 사건 보고만 있고 **현재 의식 상태 직접 표현이 없으면** → 'unknown'.\n"
        "• 질문/가정/추측/행정 전언/타 환자/모호한 상황기술만으로는 판정 금지 → 'unknown'.\n"
        "• 의식 직접 표현이 없는 경우 → 'unknown'.\n"
        "• 서로 다른 등급이 동시에 제시되거나 시점 불명확 시 → 'missing'. 시간 흐름이 분명하면 **가장 나중 시점의 직접 진술 1개**만 채택.\n"
        "• 출력은 A/V/P/U 또는 unknown/missing만 허용.\n"
        "\n"
        "— A(Alert) —\n"
        "  • 인정: '의식 명료', '의식 있다', 'alert', 'awake and oriented', 'LOC 없다'.\n"
        "  • 배제: '주취/술 드심', 'confused/혼미', 'drowsy/졸림', 'slurred/횡설수설', '답변이 느림/단답', '정신 오락가락' 등 **불완전 소통**이 언급되면 A 불가.\n"
        "  • **직접 의식 표현 없이** '말한다/대답한다/움직인다/바이탈 정상'만 있으면 A 금지 → 'unknown'.\n"
        "\n"
        "— V(Verbal) —\n"
        "  • 인정: **불완전 소통** 직접 언급(예: '횡설수설', '혼미', 'drowsy', 'confused') 및 '주취', '음주' 상태.\n"
        "  • 배제: 단순 '말한다/대답한다' **뿐**이면 V 금지 → 'unknown'.\n"
        "\n"
        "— P(Pain) —\n"
        "  • 인정: '**통증(자극)에만 반응**', '**소통 불가하나 통증엔 반응**', '세미코마(통증 반응)'.\n"
        "\n"
        "— U(Unresponsive) —\n"
        "  • 인정: '의식 없음', '무반응', '깨어나지 않음', '반응 없음', 'unresponsive'.\n"
        "  • 또한 '심정지/CPR 중/제세동 중/asystole/PEA/무맥/무호흡' **현재형 직접 진술**이 있으면 U.\n"
        "\n"
        "— AVPU 자기검증(출력 직전 필수) —\n"
        "  • 선택한 등급(A/V/P/U)을 지지하는 **직접 표현 키워드**가 스크립트에 실제 존재하는지 확인.\n"
        "  • 해당 직접 표현이 없으면 **AVPU를 'unknown'으로 교체**.\n\n"

        " +【AVPU 강제 증거-게이트(최우선)】\n"
        " - AVPU를 A/V/P/U 중 하나로 출력하려면, 반드시 스크립트에서 해당 등급을 뒷받침하는 문구를 'evidence_avpu'에 **원문 그대로 1개 이상** 복사해 넣어야 한다.\n"
        " - evidence_avpu가 비어있거나, 아래 허용 키워드가 evidence_avpu에 포함되지 않으면 AVPU는 무조건 'unknown'으로 출력한다(예외 없음).\n"
        " - '말한다/대답한다/움직인다/눈 뜬다/숨 쉰다/맥박 있다/동공 반응/바이탈 정상'만으로는 어떤 경우에도 A/V/P/U를 선택할 수 없고, 반드시 unknown이다.\n"
        " [evidence_avpu 허용 키워드] \n"
        " - A: '의식 명료', '의식 있다', '정신이 또렷', 'alert', 'awake and oriented', 'LOC 없다' \n"
        " - V: '혼미', '기면', '졸림', 'drowsy', 'confused', '횡설수설','주취', 'slurred' \n"
        " - P: '통증에만 반응', '통증 자극', 'pain' \n\n"
        "• 의식/멘탈에 대한 **직접 언급 자체가 전혀 없으면** → 'unknown'.\n"
        

        "【Intentionality_code — 필수 세밀 규칙】\n"
        "• '사고/실수/넘어짐/미끄러짐/교통사고' 등 → '1'(accidental)\n"
        "• '자해/자살 시도/본인이 일부러' 등 → '2'(self-harm)\n"
        "• '폭행/맞았다/구타/싸움/칼에 찔림(타인 가해)' 등 → '3'(violence)\n"
        "• 직접적인 언급이 없으면 → 'unknown'\n\n"

        "【Injury mechanism_code — 결정/시간 규칙 포함】\n"
        "  · 1(car accident): 승용차/택시/버스/트럭/자동차(탑승자) 사고\n"
        "  · 2(bike): 자전거\n"
        "  · 3(motorcycle): 오토바이/스쿠터/이륜\n"
        "  · 4(other traffic): 보행자 사고(차에 치임/충돌), 전동킥보드/킥보드/전동, 트랙터 등 기타 교통수단\n"
        "  · 9(unspecified traffic): '교통사고'만 있고 세부(보행자, 차종) 불명\n"
        "  · 10(fall): 낙상/추락/넘어짐\n"
        "  · 11(slipped): 미끄러져 넘어짐/미끄러\n"
        "  · 12(struck): 물체/사람과 충돌·맞음(부딪힘/낙하물에 맞음/물건에 끼임/걸림/사람에게 맞음 등)\n"
        "  · 21(firearm/cut/pierced): 총/총상 + 열상, 칼·유리·날 등으로 인한 베임/자상/찔림/동물에 의해 물림/절창\n"
        "  · 30(machine): 절단/부분 절단/완전 절단, 기계/공장/산업/절단기/말림/끼임 등\n"
        "  · 40(drowning): 익수/물에 빠짐\n"
        "  · 50(choking): 이물/기도폐쇄/사레/질식\n"
        "  · 60(fire):  화재/불/열화상(뜨거운 물체/액체 포함)\n"
        "  · 88(others): 위 항목에 속하지 않는 비교통 원인이 뚜렷할 때만\n"
        "  · unknown: injury mechanism 관련 **직접 언급이 없음**(데이터 없음)\n"
        "  · missing: injury mechanism 관련 **직접 언급은 있으나** 기전이 불명/모름/충돌로 단일 코드 확정 불가\n"
        "• 시간 충돌 시: **문서상 가장 나중 사건 1개**만 기록. 순서 불명확 시 교통 하위군(1/2/3/4/9) 우선.\n\n"

        "【Type of injury_code】\n"
        "  · 1(blunt): 넘어짐/낙상/추락/교통사고 등(압궤상 포함)\n"
        "  · 2(penetrating): 날카로운 물체로 관통/깊은 자상인 경우만\n"
        "  · 3(burn): 화상/전기/화학/흡입(열)\n"
        "  · 8(기타): 절단/부분 절단, 폴리트라우마 등\n"
        "  · unknown: 손상 유형 관련 **직접 언급이 없음**(데이터 없음)\n"
        "  · missing: 손상 유형 관련 **직접 언급은 있으나** blunt/penetrating/burn/기타로 단일 분류가 불가하거나 모름/불명/충돌\n\n"

        "【Use of protective equipment_code — 착용 판정】\n"
        "• '헬멧/보호구 착용' 직접 언급 → '1', '미착용' 직접 언급 → '2', 언급 없음 → 'unknown'.\n\n"

        "【Occupation-related_code — 업무 관련】\n"
        "• '일하다/근무/작업/배달 중' 등 **직접 언급** 있을 때만 업무 관련 코드.\n"
        "• 직접적인 언급이 없으면 → 'unknown'\n\n"

        "【바이탈 규칙 — 결정형】\n"
        "• [우선순위] (1) 개별 측정실패(-1 해당 항목) > (2) 범위 유효성 검사 > (3) '-대' 표현 처리 > (4) 대본 내 마지막 수치 1회 채택.\n"
        "• [개별 측정실패 -1] 해당 바이탈에 대해 측정 시도 후 실패가 직접 언급되면 그 항목만 '-1'. 단순 '미측정/모름'은 '-1' 금지 → 'missing'.\n"
        "• [범위 유효성] 벗어나면 raw/codes 모두 'missing': SBP/DBP/PR 0–300, RR 0–99, SpO2 0–100, BT는 0 또는 20.0–45.0.\n"
        "• ['-대' 표현] 80대/120대 등 범주형 수치는 모두 'missing'.\n"
        "• [수치 선택] 각 바이탈은 **대본에서 가장 마지막에 등장한 1회 수치**만 채택.\n"
        "• [체온 보정] 체온이 한 자리수(0 제외)면 raw는 그대로, codes에서는 +30.0 보정.\n"
        "• [동기화] raw가 '-1'인 바이탈은 codes도 '-1'.\n"
        "• [언급 없음] 바이탈에 대한 언급이 없으면 'unknown'.\n\n"

        "【Accident location_code — 장소 판정】\n"
        "• 직접 장소 표현 우선. 사건 발생 장소만 선택.\n"
        "• 직접적인 언급이 없으면 → 'unknown'\n\n"
        "• 교통사고(TA)에서 '고속도로' 언급이 없으면 '일반도로'로 간주.\n\n"

        "【Route / Mode — 기본값+오버라이드(간단)】\n"
        "• (예외) Route/Mode는 직접 언급이 없어도 codes에 기본값(Route=1, Mode=1)을 채움.\n"
        "• 직접 언급 시 교체(자가/택시/승용차: Mode=6 / 경찰: Mode=4 / 항공: Mode=5 / 도보: Mode=7 / 전원: Route=2).\n"
        "• 충돌 시: 문서상 가장 나중 직접 진술 1개만.\n\n"

        "【Insurance type_code — 보험】\n"
        "• 보험 종류 직접 언급 있을 때만 해당 코드. 맥락 추정 금지. 없으면 'unknown'.\n\n"

        "【Prehospital notification status_code — 기본값】\n"
        "• (예외) 직접 언급이 없어도 본 대화는 병원전 통화 상황으로 간주: 반증 없으면 '1'(notified).\n\n"

        "【최종 매핑 검증(강제)】\n"
       "• 결정형 슬롯(Age/AVPU/심정지/바이탈)은 위 전용 규칙을 위반하면 **직접 언급 없음→'unknown'**, **직접 언급은 있으나 위반/충돌/판정 불가→'missing'**으로 교체(임의 보정/추정 금지).\n"
        "• 비결정형 슬롯은 codebook 요약 범위에서만 매핑하고, raw와 codes가 모순되면 둘 중 하나를 근거 기반으로 수정(근거 없으면 'unknown'/'missing').\n\n"

        "【출력 형식(중요)】\n"
        "{\n"
        '  "raw":   { ...21개 슬롯... },\n'
        '  "codes": { ...21개 슬롯... }\n'
        "}\n"
        "주의: 오직 JSON 객체 하나만 출력하세요."
)





# =========================
# 호출 유틸
# =========================
def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s or "{}")
    except Exception:
        m = re.search(r"\{.*\}", s or "", re.S)
        return json.loads(m.group(0)) if m else {}

def _call_chat_schema(client: OpenAI, model: str, schema: Dict[str, Any], transcript: str, prompt_text: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": transcript or ""},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0,
    )
    elapsed = time.perf_counter() - t0

    # 토큰/시간 출력 + 집계
    try:
        u = resp.usage or {}
        pt = getattr(u, "prompt_tokens", None) or u.get("prompt_tokens", 0)
        ct = getattr(u, "completion_tokens", None) or u.get("completion_tokens", 0)
        tt = getattr(u, "total_tokens", None) or u.get("total_tokens", 0)
    except Exception:
        pt = ct = tt = 0

    if VERBOSE_METRICS:
        print(f"[openai] schema  model={model}  prompt={pt}  completion={ct}  total={tt}  latency={elapsed:.2f}s")

    # 배치 집계
    METRICS["calls"] += 1
    METRICS["prompt_tokens"] += int(pt or 0)
    METRICS["completion_tokens"] += int(ct or 0)
    METRICS["total_tokens"] += int(tt or 0)
    METRICS["elapsed_s"] += elapsed

    return _safe_json(resp.choices[0].message.content)


def _call_chat_json_only(client: OpenAI, model: str, transcript: str, prompt_text: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text + "\n반드시 JSON 객체만 출력하세요."},
            {"role": "user", "content": transcript or ""},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    elapsed = time.perf_counter() - t0

    # 토큰/시간 출력 + 집계
    try:
        u = resp.usage or {}
        pt = getattr(u, "prompt_tokens", None) or u.get("prompt_tokens", 0)
        ct = getattr(u, "completion_tokens", None) or u.get("completion_tokens", 0)
        tt = getattr(u, "total_tokens", None) or u.get("total_tokens", 0)
    except Exception:
        pt = ct = tt = 0

    if VERBOSE_METRICS:
        print(f"[openai] fallback model={model}  prompt={pt}  completion={ct}  total={tt}  latency={elapsed:.2f}s")

    METRICS["calls"] += 1
    METRICS["prompt_tokens"] += int(pt or 0)
    METRICS["completion_tokens"] += int(ct or 0)
    METRICS["total_tokens"] += int(tt or 0)
    METRICS["elapsed_s"] += elapsed

    return _safe_json(resp.choices[0].message.content)


def _supports_structured(model: str) -> bool:
    m = (model or "").lower()
    return ("4o" in m) or ("4.1" in m) or ("-mini" in m)



# =========================
# 개별 정확도 출력
# =========================
# --- metrics utils ---
IGNORE_LABELS = {"unknown", "missing"}

def _norm_code_str(x) -> str:
    s = str(x).strip()
    low = s.lower()

    # 결측/유사 결측 → unknown
    if low in {"", "nan", "none", "null"}:
        return "unknown"
    if low in {"unknown", "미상", "없음"}:
        return "unknown"
    if low in {"missing", "not available"}:
        return "missing"

    # 성별 등 알파벳 코드는 대소문자 영향 제거
    if low in {"m", "male"}:
        return "m"
    if low in {"w", "female", "f"}:
        return "w"

    # 숫자 통일(1.0 → 1, 36.50 → 36.5)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        try:
            val = float(s)
            if val.is_integer():
                return str(int(val))
            # 소수점 끝 0 제거
            txt = f"{val}"
            txt = re.sub(r"(\.\d*?)0+$", r"\1", txt).rstrip(".")
            return txt
        except Exception:
            pass

    return low  # 그 외는 소문자 비교


def _print_accuracy_metrics(keys: List[str], merged_rows: List[Dict[str, Any]]) -> None:
    """
    merged_rows: extract_slots_openai에서 만드는 dict 리스트
                 각 row는 21개 키에 대해  k_truth / k_pred  컬럼을 가짐
    """
    total_all = correct_all = 0
    total_lab = correct_lab = 0

    per_key = {
        k: {"tot": 0, "ok": 0, "lab_tot": 0, "lab_ok": 0}
        for k in keys
    }

    for row in merged_rows:
        for k in keys:
            t = _norm_code_str(row.get(f"{k}_truth", "unknown"))
            p = _norm_code_str(row.get(f"{k}_pred", "unknown"))

            # 전체 기준
            per_key[k]["tot"] += 1
            total_all += 1
            if p == t:
                per_key[k]["ok"] += 1
                correct_all += 1

            # labeled-only (truth가 unknown/missing이 아닌 경우만)
            if t not in IGNORE_LABELS:
                per_key[k]["lab_tot"] += 1
                total_lab += 1
                if p == t:
                    per_key[k]["lab_ok"] += 1
                    correct_lab += 1

    acc_all = (correct_all / total_all) if total_all else 0.0
    acc_lab = (correct_lab / total_lab) if total_lab else 0.0

    print("\n[metrics] === 전체 정확도 ===")
    print(f" all-samples     : {acc_all:.3%}  ({correct_all}/{total_all})")
    print(f" labeled-only    : {acc_lab:.3%}  ({correct_lab}/{total_lab})")

    print("\n[metrics] === 슬롯(피처)별 정확도 ===")
    w = max(len(k) for k in keys) if keys else 20
    for k in keys:
        a = per_key[k]
        acc_a = (a["ok"]     / a["tot"])     if a["tot"]     else 0.0
        acc_l = (a["lab_ok"] / a["lab_tot"]) if a["lab_tot"] else 0.0
        print(f"{k:<{w}}  all={acc_a:.3%} ({a['ok']}/{a['tot']})   labeled={acc_l:.3%} ({a['lab_ok']}/{a['lab_tot']})")
    print()  # trailing newline



# =========================
# Age helpers (deterministic post-processing)
# =========================
AGE_REFERENCE_YEAR = int(os.getenv("AGE_REFERENCE_YEAR", "2025"))

def _age_label_from_text(raw_age_label: str, transcript: str, ref_year: int = AGE_REFERENCE_YEAR) -> str:
    """Return a normalized age label used for coding.
    Output examples: '39세', '30대', '0세', 'unknown'
    Priority: year-born > explicit age (세/살) > decades(대) > infant hints.
    """
    raw = unicodedata.normalize("NFKC", str(raw_age_label or "")).strip()
    txt = unicodedata.normalize("NFKC", str(transcript or "")).strip()

    # unify for regex scanning
    combined = f"{raw} \n {txt}"

    # 0) quick unknown/missing passthrough
    low = raw.strip().lower()
    if low in {"", "unknown", "missing", "nan", "none", "null"}:
        raw = ""

    # helper: remove spaces
    def _squeeze(s: str) -> str:
        return re.sub(r"\s+", "", s or "")

    # A) 'YYYY년생' or 'YY년생' (or 'YY년생' without '년', e.g., '69년생')
    m = re.findall(r"(\d{2,4})\s*년\s*생", combined)
    if not m:
        m = re.findall(r"(\d{2,4})\s*년생", combined)  # lenient
    if m:
        yy = int(m[-1])  # last mention wins
        if yy < 100:
            # dynamic split based on ref_year (e.g., 25 in 2025)
            cut = ref_year % 100
            birth = 2000 + yy if 0 <= yy <= cut else 1900 + yy
        else:
            birth = yy
        age = (ref_year - birth) + 1  # Korean age, consistent with prompt intent
        if 0 <= age <= 130:
            return f"{age}세"
        return "unknown"

    # B) explicit age: '만 39세', '39세', '39살' etc (take last)
    # strip leading '만' for matching
    comb2 = re.sub(r"\b만\s*", "", combined)
    m2 = re.findall(r"(\d{1,3})\s*(세|살)\b", comb2)
    if m2:
        n = int(m2[-1][0])
        if 0 <= n <= 130:
            return f"{n}세"
        return "unknown"

    # C) infants by months/weeks/days (map to <1 => '0세')
    if re.search(r"(\d+)\s*(개월|달|개월째|개월차|주|일)\b", combined) or re.search(r"(신생아|갓난아기|영아)", combined):
        return "0세"

    # D) decades: '30대', '30대초반', '30~40대' (only adjacent allowed -> '30대')
    t = _squeeze(combined)
    m3 = re.findall(r"(\d{2,3})대", t)
    # handle ranges like 30~40대 / 30-40대
    mr = re.findall(r"(\d{2,3})[~\-](\d{2,3})대", t)
    if mr:
        x, y = int(mr[-1][0]), int(mr[-1][1])
        if y == x + 10 and 10 <= x <= 110:
            return f"{x}대"
        return "unknown"
    if m3:
        x = int(m3[-1])
        if 10 <= x <= 110 and x % 10 == 0:
            return f"{x}대"
        return "unknown"

    # E) numeric-only age (rare): '39' -> '39세'
    if raw and re.fullmatch(r"\d{1,3}", _squeeze(raw)):
        n = int(_squeeze(raw))
        if 0 <= n <= 130:
            return f"{n}세"

    # F) final normalize (keep compact form) if something like '39 세'
    if raw:
        raw2 = _squeeze(re.sub(r"^\s*만\s*", "", raw))
        return raw2 or "unknown"

    return "unknown"


def _age_label_to_code(age_label: str) -> str:
    """Convert normalized age label ('N세' or 'XX대' or '0세') to fixed code."""
    s = unicodedata.normalize("NFKC", str(age_label or "")).strip()
    if s in {"unknown", "missing", ""}:
        return "unknown"

    # decades mid-point codes
    if re.fullmatch(r"\d{2,3}대", s):
        x = int(s[:-1])
        decade_map = {
            10: "4.5", 20: "6.5", 30: "8.5", 40: "10.5", 50: "12.5",
            60: "14.5", 70: "16.5", 80: "18.5", 90: "20.5", 100: "22.5", 110: "24.5",
        }
        return decade_map.get(x, "unknown")

    # N세
    m = re.fullmatch(r"(\d{1,3})세", s)
    if not m:
        m = re.fullmatch(r"(\d{1,3})살", s)
    if m:
        n = int(m.group(1))
        if n < 1:
            return "1"  # <1
        if 1 <= n <= 4:   return "2"
        if 5 <= n <= 9:   return "3"
        if 10 <= n <= 14: return "4"
        if 15 <= n <= 19: return "5"
        if 20 <= n <= 24: return "6"
        if 25 <= n <= 29: return "7"
        if 30 <= n <= 34: return "8"
        if 35 <= n <= 39: return "9"
        if 40 <= n <= 44: return "10"
        if 45 <= n <= 49: return "11"
        if 50 <= n <= 54: return "12"
        if 55 <= n <= 59: return "13"
        if 60 <= n <= 64: return "14"
        if 65 <= n <= 69: return "15"
        if 70 <= n <= 74: return "16"
        if 75 <= n <= 79: return "17"
        if 80 <= n <= 84: return "18"
        if 85 <= n <= 89: return "19"
        if 90 <= n <= 94: return "20"
        if 95 <= n <= 99: return "21"
        if 100 <= n <= 104: return "22"
        if 105 <= n <= 109: return "23"
        if 110 <= n <= 114: return "24"
        if 115 <= n <= 119: return "25"
        if n >= 120: return "26"
    return "unknown"


# =========================
# Cardiac Arrest 보조 함수
# 키워드 기반 역매핑
# =========================

def _normalize_korean_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_any_cardiac_arrest_related_evidence(transcript: str) -> bool:
    txt = _normalize_korean_text(transcript)

    positive_patterns = [
        r"심정지",
        r"\bcardiac arrest\b",
        r"\bcpa\b",
        r"\bcpr\b",
        r"심폐소생술",
        r"제세동",
        r"\brosc\b",
        r"\basystole\b",
        r"\bpea\b",
        r"무맥",
        r"맥박(이)? 없",
        r"무호흡",
        r"호흡(이)? 없",
        r"심정지(가)? 왔",
        r"심정지 상태",
        r"심정지 환자",
        r"심정지입니다",
        r"자발순환(이)? 돌아",
        r"맥박(이)? 돌아",
        r"호흡(이)? 돌아",
    ]

    return any(re.search(p, txt) for p in positive_patterns)


# =========================
# 메인 콜
# =========================
def call_openai_slotfill(
    transcript: str,
    codebook_path: str,
    model: str = "gpt-4o",
    api_key: str = None,
) -> Dict[str, Any]:
    KEYS_21, enums, cats = load_codebook(codebook_path)
    schema = build_dual_schema(KEYS_21, enums) if DUAL_OUTPUT else build_schema(KEYS_21, enums)
    client: OpenAI = get_openai_client(api_key)

    # 허용 enum 캐시 (바이탈은 enum 없음)
    schema_allowed: Dict[str, List[str]] = {}
    try:
        props = (schema.get("schema") or {}).get("properties") or {}
        code_props = (props.get("codes") or {}).get("properties") or props
        for slot_key in KEYS_21:
            prop = code_props.get(slot_key, {})
            if isinstance(prop, dict) and "enum" in prop:
                schema_allowed[slot_key] = [str(x) for x in prop["enum"]]
    except Exception:
        schema_allowed = {}

    # === 단일 호출: raw + codes 동시에 ===
    prompt_text = _prompt(KEYS_21, cats)
    try:
        resp = _call_chat_schema(client, model, schema, transcript, prompt_text)
    except Exception:
        resp = _call_chat_json_only(client, model, transcript, prompt_text)

    # (1) 먼저 모델 응답 파싱
    if DUAL_OUTPUT and isinstance(resp, dict) and "raw" in resp and "codes" in resp:
        raw_obj  = resp["raw"]   or {}
        codes_obj = resp["codes"] or {}
    else:
        raw_obj, codes_obj = {}, resp or {}

    # (2) 그 다음에 키 전처리(접미사 제거만)
    raw_obj  = { _normalize_incoming_prop_key(k): v for k, v in (raw_obj or {}).items() }
    codes_obj = { _normalize_incoming_prop_key(k): v for k, v in (codes_obj or {}).items() }


    # ----- 코드북 참조 슬롯을 스키마에 맞춰 동적으로 구성 -----
    avpu_key = "Initial AVPUa scale_code" if "Initial AVPUa scale_code" in KEYS_21 else "Initial AVPU scale_code"
    codebook_slots = {
        "Age_code",
        "Injury mechanism_code",
        "Type of injury_code",
        avpu_key,
        "Accident location_code",
        "Occupation-related_code",
    }

    # 1) raw 캐논화 (그대로 보존)
    raw_extracted: Dict[str, Any] = {}
    for k, v in (raw_obj or {}).items():
        ck = _canon_key(k, KEYS_21)
        if ck:
            raw_extracted[ck] = v
    for k in KEYS_21:
        if k not in raw_extracted:
            raw_extracted[k] = "unknown"

    # 2) codes 정규화/보정 (라벨/빈값 정리, 바이탈은 건드리지 않음)
    norm: Dict[str, Any] = {}
    for k, v in (codes_obj or {}).items():
        ck = _canon_key(k, KEYS_21)
        if ck:
            norm[ck] = v

    for k in KEYS_21:
        v = norm.get(k, "unknown")
        if v in (None, "", [], {}):
            v = "unknown"
        if k.lower() == "gender_code":
            low = str(v).strip().lower()
            if low in {"여", "여성", "woman", "female", "f", "w", "여자", "여아"}:
                v = "W"
            elif low in {"남", "남성", "man", "male", "m", "남자", "남아"}:
                v = "M"
            elif low == "":
                v = "unknown"
        norm[k] = v

    # --- cardiac arrest: only reconsider OpenAI's code "2" ---
    if "Prehospital cardiac arrest_code" in KEYS_21:
        cur_ca = str(norm.get("Prehospital cardiac arrest_code", "")).strip()

        if cur_ca in {"2", "2.0"}:
            has_related = _has_any_cardiac_arrest_related_evidence(transcript)

            if not has_related:
                norm["Prehospital cardiac arrest_code"] = "unknown"
            else:
                norm["Prehospital cardiac arrest_code"] = "2"

    # --- 선택 가드: 바이탈에서 raw가 '-1'이면 codes도 '-1'로 강제 동기화 ---
    for vk in KEYS_21:
        if _is_vital_prop(vk):
            if str(raw_extracted.get(vk, "")).strip() == "-1":
                norm[vk] = "-1"


    # 3) 라벨→코드 역매핑: 코드북 대상 슬롯만, 바이탈은 스킵
    for k in KEYS_21:
        if _is_vital_prop(k) or (k not in codebook_slots):
            continue

        allowed = schema_allowed.get(k) or enums.get(k, [])
        val = str(norm.get(k, "unknown")).strip()

        if k == "Age_code":
            # Deterministic Age coding (handles year-born, explicit age, decades, infant hints).
            age_label = _age_label_from_text(raw_extracted.get("Age_code", ""), transcript)
            norm[k] = _age_label_to_code(age_label)
            continue


        # 0) 이미 유효 코드면 손대지 않음
        if (allowed and val in allowed) or (val in {"unknown", "missing"}):
            continue

        # 1) 별칭/라벨 역매핑 준비: 문자열/리스트 모두 지원
        def _norm_label_token(s: str) -> str:
            s = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
            s = s.replace("–", "-").replace("—", "-")   # 대시 통일
            s = re.sub(r"\s+", "", s)                   # 모든 공백 제거
            if s.startswith("만"):                      # '만18세' -> '18세'
                s = s[1:]
            return s

        inv = {}
        for ccode, labels in (cats.get(k) or {}).items():
            seq = labels if isinstance(labels, list) else [labels]
            for lab in seq:
                key = _norm_label_token(lab)
                if key:  # 마지막에 등장한 별칭이 우선(덮어쓰기)
                    inv[key] = str(ccode)

        # 2) 모델이 낸 라벨/별칭을 정규화 후 코드로 역매핑
        code = inv.get(_norm_label_token(val))
        if (not code) and (k == "Age_code"):
                if re.fullmatch(r"\d{1,3}", val.strip()):
                    code = inv.get(_norm_label_token(val.strip() + "세")) or inv.get(_norm_label_token(val.strip() + "살"))

        # 3) 매핑 성공 시만 교체, 실패하면 'unknown'으로 정리(라벨 문자열 방치 금지)
        if code and ((not allowed) or (code in allowed)):
            norm[k] = code
        else:
            norm[k] = "unknown"



    # 바이탈 codes ← raw 강제 동기화는 하지 않음.
    # (프롬프트 규칙에 따라 모델이 codes에서만 체온 30대 보정을 적용해야 하므로)

    return {"final": norm, "raw": raw_extracted}

# =========================
# 배치 유틸
# =========================
def extract_slots_openai(
    transcripts: List[str],
    schema_path: str,   # codebook path
    model: str = "gpt-4o",
    api_key: str = None,
    truth_rows: List[Dict[str, Any]] | None = None,   
    write_preds: bool = True,
    truth_aliases_lower: Dict[str, List[str]] | None = None,                        
) -> List[Dict[str, Any]]:
    """
    배치 추론:
      - outputs/predictions/preds_raw_labels.xlsx : raw 라벨 표(코드북 순서에 따라)
      - outputs/predictions/raw_labels.jsonl      : raw 라벨 JSONL
      - outputs/debug/first_raw.json              : 첫 샘플 raw 스냅샷
      - (옵션) outputs/predictions/preds.xlsx     : truth + pred 병합 결과
    반환: 최종 코드화 결과(dict) 리스트
    """
    KEYS_21, _, _ = load_codebook(schema_path)

    out_final: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    merged_rows: List[Dict[str, Any]] = []  # truth+pred 저장용

    os.makedirs("outputs/debug", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    raw_jsonl_path = "outputs/predictions/raw_labels.jsonl"
    with open(raw_jsonl_path, "w", encoding="utf-8") as _fclear:
        pass

    for i, t in enumerate(transcripts or []):
        res = call_openai_slotfill(t, codebook_path=schema_path, model=model, api_key=api_key)
        final_obj = res["final"]
        raw_obj   = res["raw"]

        out_final.append(final_obj)

        # ---- raw 표/JSONL ----
        raw_row = {k: str(raw_obj.get(k, "unknown")) for k in KEYS_21}
        raw_rows.append(raw_row)

        raw_str = json.dumps(raw_obj, ensure_ascii=False)
        with open(raw_jsonl_path, "a", encoding="utf-8") as fjsonl:
            fjsonl.write(raw_str + "\n")

        if i == 0:
            with open("outputs/debug/first_raw.json","w",encoding="utf-8") as f:
                json.dump(raw_obj, f, ensure_ascii=False, indent=2)

        # ---- truth 병합(옵션) ----
        if truth_rows is not None:
            tr = truth_rows[i] if i < len(truth_rows) else {}
            merged = {}
            for k in KEYS_21:
                # pred
                merged[f"{k}_pred"] = str(final_obj.get(k, "unknown"))
                # truth (AVPU/AVPUa 교차 포함, *_truth 유무 모두 지원)
                merged[f"{k}_truth"] = str(_get_truth_value_from_row(
                    tr, k, KEYS_21, default="unknown", truth_aliases_lower=truth_aliases_lower  # ★ 추가
                ))
            merged_rows.append(merged)

    # ---- 파일 저장 ----
    if raw_rows:
        df_raw = pd.DataFrame(raw_rows)[KEYS_21]
        df_raw.to_excel("outputs/predictions/preds_raw_labels.xlsx", index=False)

    if write_preds and merged_rows:
        df_preds = pd.DataFrame(merged_rows)
        # 보기 좋게 truth 먼저, pred 나중 순서
        ordered_cols = [f"{k}_truth" for k in KEYS_21] + [f"{k}_pred" for k in KEYS_21]
        df_preds = df_preds.reindex(columns=ordered_cols)
        df_preds.to_excel("outputs/predictions/preds.xlsx", index=False)


    # 배치 요약 출력
    if VERBOSE_METRICS and METRICS["calls"] > 0:
        avg_s = METRICS["elapsed_s"] / METRICS["calls"]
        print(
            "[batch] calls={calls}  total_tokens={tt} (prompt={pt}, completion={ct})  "
            "elapsed={el:.2f}s  avg/call={avg:.2f}s".format(
                calls=METRICS["calls"],
                tt=METRICS["total_tokens"],
                pt=METRICS["prompt_tokens"],
                ct=METRICS["completion_tokens"],
                el=METRICS["elapsed_s"],
                avg=avg_s,
            )
        )
        # 추가: truth_rows가 있을 때 슬롯별 정확도 즉시 출력
        if truth_rows:
            _print_accuracy_metrics(KEYS_21, merged_rows)

    return out_final

