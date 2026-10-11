import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Filter, History, RefreshCw, X } from 'lucide-react';

import { PredictionCard } from '../../components/PredictionCard';
import { PredictionResult } from '../../components/PredictionResult';
import { predictionApi } from '../../services/predictionApi';
import type { ModelInfo, PredictionHistoryGroup, PredictionResponse, PredictionStatus } from '../../types/prediction';

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'COMPLETED', label: 'Completed' },
  { value: 'PROCESSING', label: 'Processing' },
  { value: 'FAILED', label: 'Failed' },
];
const LIMIT = 12;

function getGroupStatus(predictions: PredictionResponse[]): PredictionStatus {
  if (predictions.some((item) => item.status === 'PROCESSING')) return 'PROCESSING';
  if (predictions.some((item) => item.status === 'UPLOADED')) return 'UPLOADED';
  if (predictions.every((item) => item.status === 'FAILED')) return 'FAILED';
  return 'COMPLETED';
}

function HistoryDetail({ group, models, onClose }: {
  group: PredictionHistoryGroup;
  models: ModelInfo[];
  onClose: () => void;
}) {
  const [selectedResultId, setSelectedResultId] = useState(group.predictions[0]?.id ?? '');
  const selectedResult = group.predictions.find((item) => item.id === selectedResultId) ?? group.predictions[0];
  const modelByMethod = useMemo(() => new Map(models.map((model) => [model.id, model])), [models]);
  const [selectedBackbone, setSelectedBackbone] = useState(
    group.predictions[0]?.method?.startsWith('resnet50-') ? 'resnet50' : 'clip',
  );
  const availableResults = group.predictions.filter(
    (prediction) => modelByMethod.get(prediction.method ?? 'all')?.backbone === selectedBackbone,
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!selectedResult) return null;
  return (
    <div role="presentation" onMouseDown={onClose} style={{ position: 'fixed', inset: 0, zIndex: 200, display: 'grid', placeItems: 'center', padding: 24, background: 'rgba(15, 23, 42, .62)', backdropFilter: 'blur(4px)' }}>
      <section role="dialog" aria-modal="true" aria-label="Prediction details" onMouseDown={(event) => event.stopPropagation()} style={{ width: 'min(760px, 100%)', maxHeight: 'calc(100vh - 48px)', overflowY: 'auto', padding: 18, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg)', boxShadow: 'var(--shadow-lg)' }}>
        <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, marginBottom: 16 }}>
          <div>
            <span style={{ color: 'var(--color-text-muted)', fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>Classification details</span>
            <h2 style={{ marginTop: 3, fontSize: 20 }}>{group.batch_id ? 'All-model comparison' : 'Prediction result'}</h2>
          </div>
          <button className="btn btn-ghost" style={{ width: 38, height: 38, padding: 0 }} onClick={onClose} aria-label="Close details"><X size={18} /></button>
        </header>

        <div style={{ width: '100%', height: 'min(45vh, 420px)', padding: 8, borderRadius: 8, marginBottom: 16, background: 'var(--color-surface-2)' }}>
          <img src={group.image_key} alt="Classified pet" style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain', borderRadius: 6 }} />
        </div>

        {group.batch_id && group.predictions.length > 1 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12, marginBottom: 16 }}>
            <label style={{ display: 'grid', gap: 6 }}>
              <span style={{ color: 'var(--color-text-muted)', fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>Model</span>
              <select value={selectedBackbone} onChange={(event) => {
                const backbone = event.target.value as 'clip' | 'resnet50';
                const firstResult = group.predictions.find((prediction) => modelByMethod.get(prediction.method ?? 'all')?.backbone === backbone);
                setSelectedBackbone(backbone);
                if (firstResult) setSelectedResultId(firstResult.id);
              }} style={{ height: 42, padding: '0 12px', color: 'var(--color-text)', border: '1px solid var(--color-border)', borderRadius: 6, background: 'var(--color-surface-2)', font: '500 13px var(--font-sans)' }}>
                <option value="clip">CLIP</option>
                <option value="resnet50">ResNet-50</option>
              </select>
            </label>
            <label style={{ display: 'grid', gap: 6 }}>
              <span style={{ color: 'var(--color-text-muted)', fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>Adapter</span>
              <select value={selectedResult.id} onChange={(event) => setSelectedResultId(event.target.value)} style={{ height: 42, padding: '0 12px', color: 'var(--color-text)', border: '1px solid var(--color-border)', borderRadius: 6, background: 'var(--color-surface-2)', font: '500 13px var(--font-sans)' }}>
                {availableResults.map((prediction) => (
                  <option key={prediction.id} value={prediction.id}>{modelByMethod.get(prediction.method ?? 'all')?.name ?? prediction.method ?? 'Unknown model'} - {prediction.status}</option>
                ))}
              </select>
            </label>
          </div>
        )}
        <PredictionResult prediction={selectedResult} />
      </section>
    </div>
  );
}

export function HistoryPage() {
  const [groups, setGroups] = useState<PredictionHistoryGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState('');
  const [selectedBackbone, setSelectedBackbone] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('');
  const [page, setPage] = useState(1);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedGroup, setSelectedGroup] = useState<PredictionHistoryGroup | null>(null);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await predictionApi.listPredictionHistory({
        status: selectedStatus || undefined,
        backbone: selectedBackbone || undefined,
        method: selectedMethod || undefined,
        page,
        limit: LIMIT,
      });
      setGroups(data.items);
      setTotal(data.total);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, [selectedStatus, selectedBackbone, selectedMethod, page]);

  useEffect(() => { predictionApi.listModels().then(setModels).catch(() => setModels([])); }, []);
  useEffect(() => {
    // Synchronize history with the current server-side filters and page.
    // oxlint-disable-next-line react/set-state-in-effect
    fetchPredictions();
  }, [fetchPredictions]);

  const pageCount = Math.max(1, Math.ceil(total / LIMIT));
  const filterStyle: React.CSSProperties = { minWidth: 160, padding: '10px 14px', color: 'var(--color-text)', font: '500 14px var(--font-sans)', border: '1px solid var(--color-border)', borderRadius: 8, outline: 'none', background: 'var(--color-surface-2)', cursor: 'pointer' };

  return (
    <div style={{ minHeight: '100vh', width: '100%', maxWidth: 1100, margin: '0 auto', padding: '40px 24px 60px' }}>
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}><History size={22} color="var(--color-accent)" /><h1 style={{ fontSize: 28, fontWeight: 800 }}>Classification History</h1></div>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 15 }}>{loading ? 'Loading...' : `${total} classification requests`}</p>
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 28, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Filter size={14} color="var(--color-text-muted)" /><span style={{ fontSize: 13, color: 'var(--color-text-muted)', fontWeight: 600 }}>Filter:</span></div>
        <select value={selectedStatus} onChange={(event) => { setSelectedStatus(event.target.value); setPage(1); }} style={filterStyle}>{STATUS_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
        <select value={selectedBackbone} onChange={(event) => { setSelectedBackbone(event.target.value); setSelectedMethod(''); setPage(1); }} style={filterStyle}><option value="">All Architectures</option><option value="clip">CLIP</option><option value="resnet50">ResNet-50</option></select>
        <select value={selectedMethod} onChange={(event) => { setSelectedMethod(event.target.value); setPage(1); }} style={filterStyle}>
          <option value="">All Adapters</option>
          {models.filter((model) => !selectedBackbone || model.backbone === selectedBackbone).map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
        </select>
        <button className="btn btn-ghost" onClick={fetchPredictions} style={{ marginLeft: 'auto' }}><RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh</button>
      </div>

      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>{Array.from({ length: 6 }).map((_, index) => <div key={index} className="skeleton" style={{ height: 340, borderRadius: 8 }} />)}</div>
      ) : error ? (
        <div style={{ padding: '60px 20px', color: 'var(--color-error)', textAlign: 'center' }}><p>{error}</p><button className="btn btn-primary" onClick={fetchPredictions} style={{ marginTop: 20 }}>Retry</button></div>
      ) : groups.length === 0 ? (
        <div style={{ padding: '80px 20px', color: 'var(--color-text-muted)', textAlign: 'center' }}><History size={48} color="rgba(139,92,246,0.3)" style={{ margin: '0 auto 16px' }} /><p style={{ marginBottom: 8 }}>No predictions yet</p><p style={{ fontSize: 14, color: 'var(--color-text-faint)' }}>Upload an image on the Home page to get started</p></div>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
            {groups.map((group) => {
              const cardPrediction = { ...group.predictions[0], status: getGroupStatus(group.predictions) };
              return <PredictionCard key={group.group_id} prediction={cardPrediction} batchSize={group.batch_id ? group.predictions.length : 1} batchResults={group.predictions} onClick={() => setSelectedGroup(group)} />;
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, marginTop: 32, alignItems: 'center' }}>
            <button className="btn btn-ghost" onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={page === 1}><ChevronLeft size={16} /> Previous</button>
            <span style={{ color: 'var(--color-text-muted)', fontSize: 14 }}>Page {page} of {pageCount}</span>
            <button className="btn btn-ghost" onClick={() => setPage((value) => value + 1)} disabled={page >= pageCount}>Next <ChevronRight size={16} /></button>
          </div>
        </>
      )}

      {selectedGroup && <HistoryDetail group={selectedGroup} models={models} onClose={() => setSelectedGroup(null)} />}
    </div>
  );
}
