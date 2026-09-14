import axios from 'axios';
import type {
  PredictionCreateResponse,
  PredictionResponse,
  BreedResponse,
  SpeciesResponse,
  ModelMethod,
  ModelInfo,
} from '../types/prediction';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

export const predictionApi = {
  /**
   * Upload an image and create a new prediction record.
   * Returns immediately with id + status=UPLOADED.
   */
  createPrediction: async (file: File, method: ModelMethod = 'coop'): Promise<PredictionCreateResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post<PredictionCreateResponse>(`/predictions?method=${method}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  /**
   * Get a single prediction by its UUID.
   */
  getPrediction: async (id: string): Promise<PredictionResponse> => {
    const { data } = await api.get<PredictionResponse>(`/predictions/${id}`);
    return data;
  },

  /**
   * List predictions with optional filters and pagination.
   * Backend returns a plain array — we wrap it into a paginated shape on the client side.
   */
  listPredictions: async (params?: {
    breed_id?: number;
    status?: string;
    method?: string;
    skip?: number;
    limit?: number;
  }): Promise<PredictionResponse[]> => {
    const { data } = await api.get<PredictionResponse[]>('/predictions', { params });
    return data;
  },

  /**
   * List available classification models
   */
  listModels: async (): Promise<ModelInfo[]> => {
    const { data } = await api.get<ModelInfo[]>('/models');
    return data;
  },
};

export const breedApi = {
  /**
   * List all breeds, optionally filtered by species name.
   */
  listBreeds: async (species?: string): Promise<BreedResponse[]> => {
    const { data } = await api.get<BreedResponse[]>('/breeds', {
      params: species ? { species } : undefined,
    });
    return data;
  },

  /**
   * List all species (Cat / Dog).
   */
  listSpecies: async (): Promise<SpeciesResponse[]> => {
    const { data } = await api.get<SpeciesResponse[]>('/species');
    return data;
  },
};
