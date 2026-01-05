from __future__ import annotations 

import os 
from pathlib import Path 

from dotenv import load_dotenv

load_dotenv() #env 파일 로드

# 어떤 LLM 제공자를 사용할지 설정
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").strip().lower()
if LLM_PROVIDER not in ("claude", "gemini"):
    raise ValueError("LLM_PROVIDER must be either 'claude' or 'gemini'")


SOURCE_DIR = Path("docs/source") # 문서 소스 디렉토리
INDEX_DIR = Path("data/faiss_index") #벡터 인덱스 디렉토리

# 임베딩 모델 이름 설정 - sentence-transformers/all-MiniLM-L6-v2 사용
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest") # Claude 모델 이름 설정
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") # Gemini 모델 이름 설정

#Retriever 설정
TOP_K = int(os.getenv("TOP_K", "4")) # 검색할 상위 문서 수

