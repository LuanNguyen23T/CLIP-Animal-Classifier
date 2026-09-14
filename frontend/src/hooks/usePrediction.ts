import { useState, useEffect, useRef, useCallback } from 'react';
import { predictionApi } from '../services/predictionApi';
import type { PredictionResponse, PredictionStatus } from '../types/prediction';

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES: PredictionStatus[] = ['COMPLETED', 'FAILED'];

interface UsePredictionReturn {
  prediction: PredictionResponse | null;
  isPolling: boolean;
  error: string | null;
  startPolling: (id: string) => void;
  stopPolling: () => void;
  reset: () => void;
}

/**
 * Hook that polls GET /predictions/{id} every 2s until
 * status becomes COMPLETED or FAILED.
 */
export function usePrediction(): UsePredictionReturn {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const predictionIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPolling(false);
  }, []);

  const fetchPrediction = useCallback(async (id: string) => {
    try {
      const result = await predictionApi.getPrediction(id);
      setPrediction(result);

      if (TERMINAL_STATUSES.includes(result.status)) {
        stopPolling();
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to fetch prediction';
      setError(message);
      stopPolling();
    }
  }, [stopPolling]);

  const startPolling = useCallback((id: string) => {
    // Stop any existing poll
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    predictionIdRef.current = id;
    setPrediction(null);
    setError(null);
    setIsPolling(true);

    // Fetch immediately
    fetchPrediction(id);

    // Then poll every 2s
    intervalRef.current = setInterval(() => {
      fetchPrediction(id);
    }, POLL_INTERVAL_MS);
  }, [fetchPrediction]);

  const reset = useCallback(() => {
    stopPolling();
    setPrediction(null);
    setError(null);
    predictionIdRef.current = null;
  }, [stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return { prediction, isPolling, error, startPolling, stopPolling, reset };
}
