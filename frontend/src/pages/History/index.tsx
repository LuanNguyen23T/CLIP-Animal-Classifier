import { useState, useEffect, useCallback } from 'react';
import { Filter, RefreshCw, History } from 'lucide-react';
import { PredictionCard } from '../../components/PredictionCard';
import { predictionApi, breedApi } from '../../services/predictionApi';
import type { PredictionResponse, SpeciesResponse } from '../../types/prediction';

const STATUS_OPTIONS = [
  { value: '', label: 'Tất cả trạng thái' },
  { value: 'COMPLETED', label: 'Hoàn thành' },
  { value: 'PROCESSING', label: 'Đang xử lý' },
  { value: 'FAILED', label: 'Thất bại' },
];

const LIMIT = 12;

export function HistoryPage() {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [species, setSpecies] = useState<SpeciesResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedStatus, setSelectedStatus] = useState('');
  const [page, setPage] = useState(1);

  const fetchPredictions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await predictionApi.listPredictions({
        status: selectedStatus || undefined,
        skip: (page - 1) * LIMIT,
        limit: LIMIT,
      });
      setPredictions(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, [selectedStatus, page]);

  // Load species once (for display only)
  useEffect(() => {
    breedApi.listSpecies().then(setSpecies).catch(console.error);
  }, []);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  const filterSelectStyle: React.CSSProperties = {
    background: 'var(--color-surface-2)',
    border: '1px solid var(--color-border)',
    borderRadius: 10,
    padding: '10px 14px',
    color: 'var(--color-text)',
    fontSize: 14,
    cursor: 'pointer',
    outline: 'none',
    minWidth: 160,
    fontFamily: 'var(--font-sans)',
  };

  const hasNextPage = predictions.length === LIMIT;

  return (
    <div style={{
      minHeight: '100vh',
      padding: '40px 24px 60px',
      maxWidth: 1100,
      margin: '0 auto',
      width: '100%',
    }}>
      {/* Page Header */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <History size={22} color="var(--color-accent)" />
          <h1 style={{ fontSize: 28, fontWeight: 800 }}>Lịch sử phân loại</h1>
        </div>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 15 }}>
          {loading ? 'Đang tải...' : `${predictions.length} predictions (trang ${page})`}
        </p>
        {/* Unused but loaded: species info available */}
        {species.length > 0 && (
          <span style={{ display: 'none' }}>{species.map(s => s.name).join(',')}</span>
        )}
      </div>

      {/* Filters */}
      <div style={{
        display: 'flex',
        gap: 12,
        alignItems: 'center',
        marginBottom: 28,
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Filter size={14} color="var(--color-text-muted)" />
          <span style={{ fontSize: 13, color: 'var(--color-text-muted)', fontWeight: 600 }}>Lọc:</span>
        </div>

        {/* Status filter */}
        <select
          value={selectedStatus}
          onChange={(e) => { setSelectedStatus(e.target.value); setPage(1); }}
          style={filterSelectStyle}
          id="filter-status"
        >
          {STATUS_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>

        {/* Refresh */}
        <button
          className="btn btn-ghost"
          onClick={fetchPredictions}
          style={{ marginLeft: 'auto' }}
          id="refresh-btn"
        >
          <RefreshCw size={14} />
          Làm mới
        </button>
      </div>

      {/* Content */}
      {loading ? (
        /* Skeleton grid */
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 16,
        }}>
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 180, borderRadius: 'var(--radius-lg)' }} />
          ))}
        </div>
      ) : error ? (
        <div style={{
          textAlign: 'center',
          padding: '60px 20px',
          color: 'var(--color-error)',
        }}>
          <p style={{ fontSize: 16 }}>⚠ {error}</p>
          <button className="btn btn-primary" onClick={fetchPredictions} style={{ marginTop: 20 }}>
            Thử lại
          </button>
        </div>
      ) : predictions.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '80px 20px',
          color: 'var(--color-text-muted)',
        }}>
          <History size={48} color="rgba(139,92,246,0.3)" style={{ margin: '0 auto 16px' }} />
          <p style={{ fontSize: 16, marginBottom: 8 }}>Chưa có predictions nào</p>
          <p style={{ fontSize: 14, color: 'var(--color-text-faint)' }}>
            Upload ảnh ở trang Home để bắt đầu
          </p>
        </div>
      ) : (
        <>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 16,
          }}>
            {predictions.map((p) => (
              <PredictionCard key={p.id} prediction={p} />
            ))}
          </div>

          {/* Pagination */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: 8,
            marginTop: 32,
            alignItems: 'center',
          }}>
            <button
              className="btn btn-ghost"
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              id="prev-page-btn"
            >
              ← Trước
            </button>

            <span style={{ color: 'var(--color-text-muted)', fontSize: 14, padding: '0 16px' }}>
              Trang {page}
            </span>

            <button
              className="btn btn-ghost"
              onClick={() => setPage(p => p + 1)}
              disabled={!hasNextPage}
              id="next-page-btn"
            >
              Sau →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
