from __future__ import annotations

from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
except Exception:
    pass

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

import config


def build_llm():
    """LLM_PROVIDER 값에 따라 Claude 또는 Gemini Chat 모델을 만든다."""
    if config.LLM_PROVIDER == "claude":
        # .env에 ANTHROPIC_API_KEY가 있어야 함
        return ChatAnthropic(model=config.CLAUDE_MODEL, temperature=0)
    if config.LLM_PROVIDER == "gemini":
        # .env에 GOOGLE_API_KEY가 있어야 함
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0)

    # config.py에서 이미 검증하지만 혹시 몰라 방어
    raise ValueError("Unknown LLM_PROVIDER")


def format_context(docs) -> str:
    """검색된 문서 chunk들을 LLM에 넣기 좋은 문자열로 합친다."""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] SOURCE: {src}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    # 임베딩 객체 (ingest.py와 반드시 같은 모델명)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # FAISS 로드
    # 저장/로드 과정에서 pickle이 쓰일 수 있어 allow_dangerous_deserialization 필요할 때가 있음
    vectorstore = FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    # 사용자 질문이 들어오면 가장 유사한 문서 청크(조각)을 TOP_K개 검색해서 찾아오는 역할
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    # LLM 준비
    llm = build_llm() # LLM_PROVIDER에 따라 Claude 또는 Gemini Chat 모델 생성
    print(f"[chat] provider={config.LLM_PROVIDER} k={config.TOP_K}")
    print("[chat] 'exit' 입력해서 종료\n")

    # 대화 루프
    while True:
        question = input("Q> ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        # 관련 문서 검색
        docs = retriever.invoke(question) # 검색된 문서 청크 리스트
        context = format_context(docs) # 문서 청크들을 LLM에 넣기 좋은 문자열로 합침

        # 프롬프트: "근거 기반" 강제 (핵심 규칙만 간단히)
        prompt = f"""당신은 유능하고 친절한 어시스턴트입니다.
        
        반드시 아래 제공된 [참고 문헌]의 내용만을 바탕으로 답변하세요.
        만약 질문에 대한 답이 [참고 문헌]에 없다면, "제공된 문서에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
        답변이 끝나면 반드시 답변에 사용된 [출처]의 파일명을 나열하세요.

        CONTEXT:
        {context}

        QUESTION:
        {question}
        """

        # 모델 호출 (반환 타입은 모델별로 다를 수 있어 content 처리)
        result: Any = llm.invoke(prompt)
        answer = getattr(result, "content", result)

        print("\nA>")
        print(answer)

        # 디버깅/학습용: 어떤 chunk를 썼는지 바로 확인
        print("\n[retrieved sources]")
        for d in docs:
            print(" -", d.metadata.get("source", "unknown"))
        print()


if __name__ == "__main__":
    main()