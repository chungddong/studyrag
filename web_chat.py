from __future__ import annotations

import asyncio
from typing import Any

import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

import config


def build_llm():
    """config.LLM_PROVIDER에 따라 Claude 또는 Gemini 모델을 선택한다."""

    if config.LLM_PROVIDER == "claude":
        # .env에 ANTHROPIC_API_KEY가 있어야 함
        return ChatAnthropic(model=config.CLAUDE_MODEL, temperature=0)

    if config.LLM_PROVIDER == "gemini":
        # .env에 GOOGLE_API_KEY가 있어야 함
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0)

    raise ValueError("Unknown LLM_PROVIDER")


def format_context(docs) -> str:
    """검색된 문서 chunk를 LLM에 넣을 CONTEXT 문자열로 합친다."""

    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] SOURCE: {src}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


@cl.on_chat_start
async def on_chat_start() -> None:
    """채팅이 시작될 때 1번만 실행.

    왜 여기서 로드하나?
    - 매 메시지마다 인덱스를 다시 로드하면 매우 느림
    - 세션 단위로 retriever/llm을 만들어두면 응답이 빨라짐
    """

    # 1) 임베딩 객체 생성 (ingest.py와 동일 모델이어야 함)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # 2) 저장된 FAISS 인덱스 로드
    vectorstore = FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    # 3) LLM 준비 (Claude/Gemini 스위치)
    llm = build_llm()

    # 4) Chainlit 세션에 보관
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("llm", llm)

    await cl.Message(
        content=(
            f"Ready. provider={config.LLM_PROVIDER}, k={config.TOP_K}\n"
            "질문을 입력하면 문서 검색 후 답변합니다."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """사용자 메시지를 받을 때마다 실행."""

    question = (message.content or "").strip()
    if not question:
        await cl.Message(content="질문을 입력해줘.").send()
        return

    retriever = cl.user_session.get("retriever")
    llm = cl.user_session.get("llm")

    # 1) 관련 문서 검색 (Runnable 기반 -> invoke 사용)
    #    retriever.invoke는 sync 함수인 경우가 많아서, asyncio.to_thread로 감싸 UI를 덜 막는다.
    docs = await asyncio.to_thread(retriever.invoke, question)
    context = format_context(docs)

    # 2) 프롬프트: 근거 기반 답변 강제
    prompt = f"""당신은 유능하고 친절한 어시스턴트입니다.

반드시 아래 제공된 [참고 문헌]의 내용만을 바탕으로 답변하세요.
만약 질문에 대한 답이 [참고 문헌]에 없다면, \"제공된 문서에서 관련 내용을 찾을 수 없습니다\"라고 답하세요.
답변이 끝나면 반드시 답변에 사용된 [출처]의 파일명을 나열하세요.

[참고 문헌]
{context}

[질문]
{question}
"""

    # 3) LLM 호출 (역시 sync일 수 있어 to_thread로 처리)
    result: Any = await asyncio.to_thread(llm.invoke, prompt)
    answer = getattr(result, "content", result)

    # 4) 응답 + 근거 표시
    sources = [d.metadata.get("source", "unknown") for d in docs]
    sources_text = "\n".join(f"- {s}" for s in sources) if sources else "- (none)"

    await cl.Message(
        content=f"{answer}\n\n---\n\n[retrieved sources]\n{sources_text}"
    ).send()
