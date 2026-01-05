from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# TextSplitter는 환경에 따라 import 경로가 달라질 수 있어 try 처리
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

# 문서 소스 디렉토리에서 .md/.txt 파일을 모두 읽어서 Document 리스트로 생성
def load_source_documents(source_dir: Path) -> list[Document]:
    """docs/source 폴더에서 .md/.txt 파일을 모두 읽어서 Document 리스트로 생성"""
    if not source_dir.exists():
        raise FileNotFoundError(f"소스 디렉토리를 찾을 수 없음: {source_dir}")

    paths = []
    paths.extend(sorted(source_dir.glob("*.md")))
    paths.extend(sorted(source_dir.glob("*.txt")))

    if not paths:
        raise FileNotFoundError(f"No .md/.txt files in {source_dir}")

    docs: list[Document] = []
    for file_path in paths:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # metadata["source"]는 나중에 근거 표시할 때 사용
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(file_path.as_posix())},
            )
        )

    print(f"[ingest] 로드된 파일: {len(docs)}")
    for d in docs:
        print(f" - {d.metadata['source']} ({len(d.page_content)} chars)")
    return docs


def main() -> None:
    # 문서 로드
    docs = load_source_documents(config.SOURCE_DIR)

    # 청킹 (검색 품질에 큰 영향) - 문서를 여러 청크(조각)로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 청크 크기, 한 조각을 최대 1000 토큰으로 분할, 한 번에 읽을 수 있는 양이 제한되어 있기 때문
        chunk_overlap=150, # 청크 간 중복 토큰 수, 문맥이 끊기는 것을 방지
    )
    chunks = splitter.split_documents(docs) # 문서를 청크 단위로 분할
    print(f"[ingest] 청크 수: {len(chunks)}") # 분할된 청크 수 출력

    # 임베딩(로컬) 생성 - 문서 청크를 벡터로 변환
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    print(f"[ingest] 사용 임베딩 모델 : {config.EMBEDDING_MODEL}")

    # FAISS 인덱스 생성 + 저장
    config.INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(config.INDEX_DIR))
    print(f"[ingest] 저장된 인덱스 경로: {config.INDEX_DIR}")


if __name__ == "__main__":
    main()