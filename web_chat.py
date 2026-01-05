from __future__ import annotations

from typing import cast

import chainlit as cl
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector retrieval (FAISS + local embeddings)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain Runnable(LCEL) building blocks
try:
    # Newer import paths
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
    from langchain_core.runnables.config import RunnableConfig
except Exception:
    # Fallbacks for older versions
    from langchain.schema import StrOutputParser  # type: ignore
    from langchain.prompts import ChatPromptTemplate  # type: ignore
    from langchain.schema.runnable import Runnable  # type: ignore
    from langchain.schema.runnable.config import RunnableConfig  # type: ignore
    from langchain.schema.runnable import RunnableLambda, RunnablePassthrough  # type: ignore

import config


def build_llm():
    """config.LLM_PROVIDER에 따라 Claude 또는 Gemini 모델을 선택한다."""

    if config.LLM_PROVIDER == "claude":
        # .env에 ANTHROPIC_API_KEY가 있어야 함
        # Chainlit 예제처럼 token streaming을 쓰기 위해 streaming=True를 시도한다.
        try:
            return ChatAnthropic(model=config.CLAUDE_MODEL, temperature=0, streaming=True)
        except TypeError:
            return ChatAnthropic(model=config.CLAUDE_MODEL, temperature=0)

    if config.LLM_PROVIDER == "gemini":
        # .env에 GOOGLE_API_KEY가 있어야 함
        try:
            return ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL, temperature=0, streaming=True
            )
        except TypeError:
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

    # 4) Prompt + Runnable(LCEL) 구성
    # Chainlit의 Steps/Trace UI는 Runnable 실행 시 발생하는 중간 이벤트를
    # cl.LangchainCallbackHandler()가 받아서 표시한다.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 유능하고 친절한 어시스턴트입니다.

반드시 아래 제공된 [참고 문헌]의 내용만을 바탕으로 답변하세요.
만약 질문에 대한 답이 [참고 문헌]에 없다면, \"제공된 문서에서 관련 내용을 찾을 수 없습니다\"라고 답하세요.
답변이 끝나면 반드시 답변에 사용된 [출처]의 파일명을 나열하세요.
""",
            ),
            (
                "human",
                """[참고 문헌]\n{context}\n\n[질문]\n{question}\n""",
            ),
        ]
    )

    # 입력은 {"question": "..."}
    # - docs: retriever(question)
    # - context: format_context(docs)
    get_question = RunnableLambda(lambda x: x["question"])
    docs_runnable = get_question | retriever
    runnable: Runnable = (
        RunnablePassthrough.assign(docs=docs_runnable)
        .assign(context=RunnableLambda(lambda x: format_context(x["docs"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5) Chainlit 세션에 보관
    cl.user_session.set("runnable", runnable)

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

    runnable = cast(Runnable, cl.user_session.get("runnable"))

    # Chainlit Steps/Trace UI를 켜는 핵심: LangchainCallbackHandler
    cb = cl.LangchainCallbackHandler()
    msg = cl.Message(content="")

    # astream으로 토큰(또는 chunk)을 스트리밍하면 UI에 타이핑처럼 표시된다.
    async for chunk in runnable.astream(
        {"question": question},
        config=RunnableConfig(callbacks=[cb]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
