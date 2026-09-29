// TypeScript interfaces matching the backend API schemas

export type PredictionStatus = 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
export type BackboneType = 'clip' | 'resnet50';
export type ModelMethod = 
  | 'all' | 'zero-shot' | 'tip-adapter' | 'tip-adapter-f' | 'clip-adapter' | 'coop'
  | 'resnet50-baseline' | 'resnet50-prototype' | 'resnet50-linear' | 'resnet50-adapter' | 'resnet50-cosine';

export interface ModelInfo {
  id: ModelMethod;
  name: string;
  description: string;
  accuracy: number;
  latency_ms: number;
  params: number;
  tag: string;
  backbone: BackboneType;
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
  batch_id: string | null;
  status: PredictionStatus;
  method: ModelMethod | null;
}

export interface PredictionResponse {
  id: string;
  batch_id: string | null;
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

export interface PredictionHistoryGroup {
  group_id: string;
  batch_id: string | null;
  image_key: string;
  created_at: string;
  predictions: PredictionResponse[];
}

export interface PredictionHistoryResponse {
  items: PredictionHistoryGroup[];
  total: number;
  page: number;
  limit: number;
}

export interface PredictionsListResponse {
  items: PredictionResponse[];
  total: number;
  page: number;
  limit: number;
}
