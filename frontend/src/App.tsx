import { useMemo, useRef, useState, type ChangeEvent } from "react";
import { askQuestion, resolveViewerUrl } from "./client";
import type { AskRequest, AskResponse, SourceChunk } from "./types";

const DEFAULT_REQUEST: AskRequest = {
  question: "Trước khi có chữ Nôm, người Việt có từng có chữ viết riêng không?",
  top_k: 10,
  pool_size: 20,
  rerank: true,
};

export default function App() {
  const [request, setRequest] = useState<AskRequest>(DEFAULT_REQUEST);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [selectedSource, setSelectedSource] = useState<SourceChunk | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [answerHighlighted, setAnswerHighlighted] = useState(false);
  const answerRef = useRef<HTMLDivElement | null>(null);
  const answerTextRef = useRef<HTMLDivElement | null>(null);
  const previewRef = useRef<HTMLDivElement | null>(null);

  const resolvedSource = useMemo(() => {
    if (selectedSource) {
      return selectedSource;
    }
    return answer?.sources?.[0] ?? null;
  }, [answer?.sources, selectedSource]);

  const previewUrl = useMemo(() => {
    if (!resolvedSource?.viewer_url) {
      return null;
    }
    const absolute = resolveViewerUrl(resolvedSource.viewer_url);
    if (!absolute) {
      return null;
    }
    try {
      const url = new URL(absolute);
      if (url.hash.length > 1) {
        const params = new URLSearchParams(url.hash.slice(1));
        params.delete("search");
        const hash = params.toString();
        url.hash = hash ? `#${hash}` : "";
      }
      return url.toString();
    } catch (_) {
      return absolute.split("#")[0];
    }
  }, [resolvedSource?.viewer_url]);

  const resolvedSourceLabel = useMemo(() => {
    if (!resolvedSource) {
      return "Chưa chọn nguồn";
    }
    const pageSuffix = resolvedSource.page_number
      ? ` – trang ${resolvedSource.page_number}`
      : "";
    return `${resolvedSource.label}${pageSuffix}`;
  }, [resolvedSource]);

  const scrollAnswerIntoView = () => {
    if (typeof window === "undefined") {
      return;
    }
    window.setTimeout(() => {
      const target = answerTextRef.current ?? answerRef.current;
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 0);
  };

  const scrollPreviewIntoView = () => {
    if (typeof window === "undefined") {
      return;
    }
    window.setTimeout(() => {
      previewRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 0);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await askQuestion(request);
      setAnswer(response);
      setSelectedSource(response.sources?.[0] ?? null);
      setExpandedKey(null);
      setAnswerHighlighted(true);
      scrollAnswerIntoView();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach backend";
      setError(message);
      setAnswer(null);
      setAnswerHighlighted(false);
    } finally {
      setLoading(false);
    }
  };

  const handleSourceClick = (source: SourceChunk) => {
    setSelectedSource(source);
    setAnswerHighlighted(false);
    scrollPreviewIntoView();
  };

  const clearAnswerHighlight = () => setAnswerHighlighted(false);

  const handleInputChange = (evt: ChangeEvent<HTMLTextAreaElement>) => {
    const value = evt.target.value;
    setRequest((prev: AskRequest) => ({ ...prev, question: value }));
    clearAnswerHighlight();
  };

  return (
    <div className="app-wrapper">
      <div className="cloud-layer" aria-hidden="true">
        <span className="cloud cloud-1" />
        <span className="cloud cloud-2" />
        <span className="cloud cloud-3" />
      </div>
      <div className="app-shell">
        <section className="hero-card">
          <h1 className="hero-title">NômSense</h1>
          <p className="hero-subtitle">
            Đối thoại với bộ sưu tập Hán Nôm – truy xuất dẫn chứng chuẩn xác và
            cảm nhận sắc thái cổ điển qua từng trang sách.
          </p>
        </section>

        <section className="panel">
          <div>
            <label htmlFor="question">Câu hỏi</label>
            <textarea
              id="question"
              value={request.question}
              onChange={handleInputChange}
              placeholder="Ví dụ: Ai là người đầu tiên được cho là làm thơ Nôm?"
            />
          </div>

          <button onClick={handleSubmit} disabled={loading}>
            {loading ? "Đang phân tích…" : "Truy vấn"}
          </button>

          {loading && (
            <div className="loading-indicator" role="status" aria-live="polite">
              <span className="spinner" aria-hidden="true" />
              <span>Đang dò tìm trong thư tàng Hán Nôm…</span>
            </div>
          )}

          {error && <p className="error-text">{error}</p>}

          {answer && (
            <div
              className={
                answerHighlighted
                  ? "answer-card answer-card--highlight"
                  : "answer-card"
              }
              ref={answerRef}
              onClick={clearAnswerHighlight}
              onFocusCapture={clearAnswerHighlight}
            >
              <h2>Kết quả</h2>
              <div className="answer-text" ref={answerTextRef}>
                {answer.answer}
              </div>
              {answer.sources && answer.sources.length > 0 && (
                <div className="citation-list">
                  <strong>Nguồn tham chiếu</strong>
                  {answer.sources.map((source: SourceChunk, idx: number) => {
                    const isActive =
                      (resolvedSource?.viewer_url === source.viewer_url &&
                        resolvedSource?.label === source.label) ||
                      selectedSource?.label === source.label;
                    const key = `${source.file_name ?? source.label}-${
                      source.page_number ?? idx
                    }`;
                    const isExpanded = expandedKey === key;
                    return (
                      <div
                        key={key}
                        className={
                          isActive ? "citation-item active" : "citation-item"
                        }
                        onClick={() => handleSourceClick(source)}
                      >
                        <div className="citation-meta">
                          <span>{source.label}</span>
                          {source.page_number && (
                            <span>p. {source.page_number}</span>
                          )}
                        </div>
                        <div className="citation-actions">
                          <button
                            type="button"
                            className="ghost-button"
                            onClick={(evt) => {
                              evt.stopPropagation();
                              clearAnswerHighlight();
                              setExpandedKey(isExpanded ? null : key);
                            }}
                          >
                            {isExpanded ? "Thu gọn" : "Xem trích đoạn"}
                          </button>
                          <button
                            type="button"
                            className="ghost-button"
                            onClick={(evt) => {
                              evt.stopPropagation();
                              clearAnswerHighlight();
                              handleSourceClick(source);
                            }}
                          >
                            Xem nhanh
                          </button>
                        </div>
                        {isExpanded && (
                          <div className="snippet">{source.text}</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="source-preview" ref={previewRef}>
                <div className="source-preview-header">
                  <span>{resolvedSourceLabel}</span>
                  {resolvedSource?.viewer_url && (
                    <button
                      type="button"
                      onClick={() => {
                        const direct = resolveViewerUrl(
                          resolvedSource.viewer_url
                        );
                        if (direct && typeof window !== "undefined") {
                          window.open(direct, "_blank", "noopener,noreferrer");
                        }
                      }}
                    >
                      Mở toàn văn
                    </button>
                  )}
                </div>
                {previewUrl ? (
                  <iframe src={previewUrl} title="Xem nhanh nguồn" />
                ) : (
                  <p>
                    Chọn một nguồn để xem nhanh. Nếu nguồn không có bản PDF nội
                    bộ, bạn vẫn có thể mở toàn văn trong tab mới.
                  </p>
                )}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
