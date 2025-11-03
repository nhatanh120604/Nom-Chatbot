export interface AskRequest {
  question: string;
  top_k?: number;
  pool_size?: number;
  temperature?: number;
  rerank?: boolean;
}

export interface SourceChunk {
  label: string;
  chapter?: string;
  file_name?: string;
  source_path?: string;
  page_number?: number;
  text: string;
  viewer_url?: string | null;
}

export interface AskResponse {
  answer: string;
  citations: string[];
  sources: SourceChunk[];
}
