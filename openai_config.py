import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    key = (api_key or "").strip()
    if not key:
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OpenAI API 키가 없습니다. .env/환경변수/--api-key 중 하나를 사용하세요.")
    return OpenAI(api_key=key)
