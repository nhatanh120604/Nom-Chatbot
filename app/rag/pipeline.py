from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote, quote_plus

try:  # Torch is optional at runtime, fall back to CPU if unavailable
    import torch
except Exception:  # pragma: no cover - torch might be missing on smaller deployments
    torch = None  # type: ignore

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import CrossEncoder

from ..schemas import SourceChunk
from ..settings import Settings

LOGGER = logging.getLogger(__name__)


BOOK_TITLE_BY_FOLDER: Dict[str, str] = {
    "Book1": "Khái luận văn tự học Chữ Nôm",
    "Book2": "Ngôn ngữ. Văn tự. Ngữ văn (Tuyển tập)",
}


def iter_document_paths(data_dir: Path, extensions: Iterable[str]) -> List[Path]:
    candidates: set[Path] = set()
    for ext in extensions:
        candidates.update(path.resolve() for path in data_dir.glob(f"**/*{ext}"))
    return sorted(candidates)


def normalise_chapter_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    cleaned = re.sub(r"_+", " ", label)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def derive_chapter_metadata(path: Path) -> Dict[str, Optional[str]]:
    stem = path.stem
    parts = stem.split(".", 1)
    chapter_index: Optional[int] = None
    chapter_title = stem.replace("_", " ").strip()

    if len(parts) == 2 and parts[0].strip().isdigit():
        chapter_index = int(parts[0].strip())
        chapter_title = parts[1].strip() or chapter_title
    elif parts[0].strip().isdigit():
        chapter_index = int(parts[0].strip())

    label = chapter_title or stem
    return {"chapter_index": chapter_index, "chapter": label}


def load_documents(data_dir: Path, extensions: Sequence[str]) -> List[Document]:
    documents: List[Document] = []
    for path in iter_document_paths(data_dir, extensions):
        chapter_meta = derive_chapter_metadata(path)
        chapter_meta["chapter"] = normalise_chapter_label(chapter_meta.get("chapter"))
        book_key = path.parent.name
        book_title = BOOK_TITLE_BY_FOLDER.get(book_key, book_key.replace("_", " "))
        loader = PyMuPDFLoader(str(path))
        for doc in loader.load():
            text = doc.page_content.strip()
            if not text:
                continue

            doc_meta = chapter_meta.copy()
            doc_meta["book_key"] = book_key
            doc_meta["book_title"] = book_title
            page = doc.metadata.get("page_number")
            if page is None:
                page = doc.metadata.get("page")
            if page is not None:
                page_number = int(page) + 1
                doc.metadata["page_number"] = page_number
                doc_meta["citation_label"] = (
                    f"{book_title} – {doc_meta['chapter']} - p.{page_number}"
                )
            else:
                doc_meta["citation_label"] = f"{book_title} – {doc_meta['chapter']}"

            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("file_name", path.name)
            doc.metadata.update({k: v for k, v in doc_meta.items() if v is not None})
            documents.append(doc)

    if not documents:
        raise ValueError(f"No supported documents were loaded from {data_dir}.")

    return documents


def unique_citations(docs: Sequence[Document]) -> List[str]:
    citations: List[str] = []
    for doc in docs:
        label = doc.metadata.get("citation_label")
        if not label:
            chapter = doc.metadata.get("chapter")
            page = doc.metadata.get("page_number")
            book_title = doc.metadata.get("book_title")
            if chapter and page:
                chapter_part = f"{chapter} - p.{page}"
            elif chapter:
                chapter_part = chapter
            else:
                chapter_part = (
                    doc.metadata.get("file_name")
                    or doc.metadata.get("source")
                    or "Unknown source"
                )
            if book_title:
                label = f"{book_title} – {chapter_part}"
            else:
                label = chapter_part
        if label not in citations:
            citations.append(label)
    return citations


def format_docs(docs: Sequence[Document]) -> str:
    formatted: List[str] = []
    for doc in docs:
        label = doc.metadata.get("citation_label")
        if not label:
            chapter = doc.metadata.get("chapter")
            page = doc.metadata.get("page_number")
            book_title = doc.metadata.get("book_title")
            if chapter and page:
                chapter_part = f"{chapter} - p.{page}"
            elif chapter:
                chapter_part = chapter
            else:
                chapter_part = (
                    doc.metadata.get("file_name")
                    or doc.metadata.get("source")
                    or "Unknown source"
                )
            if book_title:
                label = f"{book_title} – {chapter_part}"
            else:
                label = chapter_part
        formatted.append(f"Source: {label}\n{doc.page_content}")
    return "\n\n".join(formatted) if formatted else "No supporting context retrieved."


def rerank_documents(
    question: str, docs: Sequence[Document], reranker: CrossEncoder, top_k: int
) -> List[Document]:
    if not docs:
        return []
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_env()

        self.device = settings.device or self._default_device()
        LOGGER.info("Using device %s for embeddings and reranker", self.device)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"device": self.device},
        )
        self.reranker = CrossEncoder(settings.rerank_model, device=self.device)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.llm = ChatOpenAI(model=settings.chat_model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
   Bạn là trợ lý hỏi–đáp chỉ sử dụng NGỮ CẢNH (các đoạn trích đã truy xuất).
YÊU CẦU:
Trả lời theo cấu trúc dưới như sau:

Kết luận: kết luận tóm tắt ở đây
(Xuống dòng ở đây)
Giải thích: giải thích và minh chứng chữ Nôm ở đây

Các điểm cần nhớ:
NẾU CHỮ NÔM CÓ TRONG NGỮ CẢNH VÀ LIÊN QUAN TỚI CÂU HỎI: PHẢI COPY VÀO CÂU TRẢ LỜI KÈM THEO GIẢI THÍCH.
Nếu trong NGỮ CẢNH có chữ Nôm, bạn phải luôn ưu tiên sử dụng chữ Nôm và tiếng Việt phối hợp với nhau trong câu trả lời, ví dụ  trả lời tiếng Việt + chữ Nôm làm  ví dụ, lấy các chữ Nôm từ ngữ cảnh để trả lời. Đặc biệt nếu trong NGỮ CẢNH có chữ Nôm liên quan đến câu hỏi, bạn cần trả lời chi tiết hơn, có thể liệt kê các ví dụ cụ thể từ ngữ cảnh (chữ Nôm).
Nếu cùng đoạn chứa cả chữ Quốc Ngữ và chữ Nôm, trình bày cả hai dạng (Quốc Ngữ trước, chữ Nôm trong ngoặc
2) Nếu NGỮ CẢNH thiếu/không liên quan, trả lời: “Không tìm thấy thông tin này trong sách/nguồn đã cho.”
3) Không suy diễn, không đưa kiến thức ngoài NGỮ CẢNH.
4) Luôn giữ nguyên thuật ngữ gốc khi cần.

Đây là  ví dụ 1 câu hỏi và câu trả lời mẫu:
Câu hỏi : Chứng tích sớm nhất về chữ Nôm là gì?
Trả lời:
Kết luận: Là các bia đá thời Lý
Giải thích:
Bia mộ bà họ Lê (1174) có chữ Nôm mượn Hán và chữ Nôm tự tạo.
Bia chùa Tháp Miếu (1210) có hơn hai chục chữ Nôm, như:
𥺹 oản (bộ mễ + uyển),
䊷 chài (bộ mịch + tài),
土而 nhe (bộ thổ + nhi),
񢂞 bơi (bộ thuỷ + bi).
""".strip(),
                ),
                ("user", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        self._vectorstore: Optional[Chroma] = None

    def _default_device(self) -> str:
        if self.settings.device:
            return self.settings.device
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
        return "cpu"

    def has_persisted_index(self) -> bool:
        directory = self.settings.resolved_persist_dir
        return directory.exists() and any(directory.iterdir())

    def load_source_documents(self) -> List[Document]:
        return load_documents(self.settings.resolved_data_dir, (".pdf",))

    def build_or_load_vectorstore(self, force_rebuild: bool = False) -> Chroma:
        if self._vectorstore is not None and not force_rebuild:
            return self._vectorstore

        persist_directory = str(self.settings.resolved_persist_dir)
        if not force_rebuild and self.has_persisted_index():
            LOGGER.info("Loading existing Chroma index from %s", persist_directory)
            self._vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
            return self._vectorstore

        LOGGER.info("Building new Chroma index at %s", persist_directory)
        documents = self.load_source_documents()
        chunks = self.splitter.split_documents(documents)
        self._vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )
        LOGGER.info(
            "Indexed %s chunks from %s source passages",
            len(chunks),
            len(documents),
        )
        return self._vectorstore

    def ensure_vectorstore(self, force_rebuild: bool = False) -> Chroma:
        return self.build_or_load_vectorstore(force_rebuild=force_rebuild)

    def ingest(self, force_rebuild: bool = False) -> None:
        self.ensure_vectorstore(force_rebuild=force_rebuild)

    def _build_source_payload(self, doc: Document) -> SourceChunk:
        book_title = doc.metadata.get("book_title")
        label = doc.metadata.get("citation_label")
        if not label:
            chapter = doc.metadata.get("chapter")
            page = doc.metadata.get("page_number")
            if chapter and page:
                chapter_part = f"{chapter} - p.{page}"
            elif chapter:
                chapter_part = chapter
            else:
                chapter_part = (
                    doc.metadata.get("file_name")
                    or doc.metadata.get("source")
                    or "Unknown source"
                )
            label = f"{book_title} – {chapter_part}" if book_title else chapter_part
        page_number = doc.metadata.get("page_number")
        chapter = doc.metadata.get("chapter")
        file_name = doc.metadata.get("file_name")
        source_path = doc.metadata.get("source")
        viewer_url = self._build_viewer_url(
            file_name=file_name, page_number=page_number, snippet=doc.page_content
        )
        return SourceChunk(
            label=label,
            page_number=page_number,
            chapter=chapter,
            book_title=book_title,
            file_name=file_name,
            source_path=source_path,
            text=doc.page_content,
            viewer_url=viewer_url,
        )

    def _build_viewer_url(
        self, *, file_name: Optional[str], page_number: Optional[int], snippet: str
    ) -> Optional[str]:
        if not self.settings.serve_docs or not file_name:
            return None
        base = self.settings.docs_mount_path.rstrip("/") or "/docs"
        url = f"{base}/{quote(file_name)}"
        fragments: List[str] = []
        if page_number is not None:
            fragments.append(f"page={page_number}")
        if snippet:
            search_term = quote_plus(snippet[:120])
            fragments.append(f"search={search_term}")
        if fragments:
            return f"{url}#{'&'.join(fragments)}"
        return url

    def ask(
        self,
        *,
        question: str,
        additional_context: Optional[str] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
        temperature: Optional[float] = None,
        rerank: bool = True,
    ) -> Dict[str, object]:
        vectorstore = self.ensure_vectorstore()
        k = pool_size or self.settings.retriever_k
        chosen_top_k = top_k or self.settings.rerank_top_k
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        candidate_docs = retriever.invoke(question)
        if rerank:
            docs = rerank_documents(
                question, candidate_docs, self.reranker, chosen_top_k
            )
        else:
            docs = list(candidate_docs[:chosen_top_k])

        context_text = format_docs(docs)
        if additional_context:
            extra = additional_context.strip()
            context_text = f"{extra}\n\n{context_text}" if context_text else extra

        previous_temperature = self.llm.temperature
        if temperature is not None:
            self.llm.temperature = temperature
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(context=context_text, question=question)
            )
        finally:
            if temperature is not None:
                self.llm.temperature = previous_temperature

        answer = response.content.strip()
        citations = unique_citations(docs)
        sources = [self._build_source_payload(doc) for doc in docs]
        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
        }
