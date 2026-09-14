// TypeScript interfaces matching the backend API schemas

export type PredictionStatus = 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

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
}

export interface PredictionResponse {
  id: string;
  image_key: string;
  predicted_breed: BreedResponse | null;
  score: number | null;
  top_k_results: TopKResult[] | null;
  model_version: string | null;
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
