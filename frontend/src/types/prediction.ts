// TypeScript interfaces matching the backend API schemas

export type PredictionStatus = 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
export type ModelMethod = 'zero-shot' | 'tip-adapter-f' | 'clip-adapter' | 'coop';

export interface ModelInfo {
  id: ModelMethod;
  name: string;
  description: string;
  accuracy: number;
  latency_ms: number;
  params: number;
  tag: string;
}

export interface SpeciesResponse {
  id: number;
  name: string;
}

export interface BreedResponse {
  id: number;
  name: string;
  species_id: number;
  species: SpeciesResponse;
}

export interface TopKResult {
  breed_id: number;
  breed_name: string;
  species: string;
  score: number;
}

export interface PredictionCreateResponse {
  id: string;
  status: PredictionStatus;
  method: ModelMethod | null;
}

export interface PredictionResponse {
  id: string;
  image_key: string;
  predicted_breed: BreedResponse | null;
  score: number | null;
  top_k_results: TopKResult[] | null;
  model_version: string | null;
  method: ModelMethod | null;
  processing_time?: number;
  status: PredictionStatus;
  error_message: string | null;
  created_at: string;
}

export interface PredictionsListResponse {
  items: PredictionResponse[];
  total: number;
  page: number;
  limit: number;
}
