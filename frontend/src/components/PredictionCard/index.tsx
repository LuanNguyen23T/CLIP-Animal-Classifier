import { CheckCircle, Clock, XCircle, Loader2, Cat, Dog } from 'lucide-react';
import type { PredictionResponse } from '../../types/prediction';

interface PredictionCardProps {
  prediction: PredictionResponse;
  onClick?: () => void;
}

const STATUS_CONFIG = {
  UPLOADED: { icon: <Clock size={12} />, className: 'badge badge-uploaded', label: 'Uploaded' },
  PROCESSING: { icon: <Loader2 size={12} />, className: 'badge badge-processing', label: 'Processing' },
  COMPLETED: { icon: <CheckCircle size={12} />, className: 'badge badge-completed', label: 'Completed' },
  FAILED: { icon: <XCircle size={12} />, className: 'badge badge-failed', label: 'Failed' },
};

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

export function PredictionCard({ prediction, onClick }: PredictionCardProps) {
  const { status, predicted_breed, score, created_at } = prediction;
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.UPLOADED;
  const breedName = predicted_breed?.name?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) ?? '—';
  const speciesName = predicted_breed?.species?.name ?? '';

  return (
    <div
      className="glass-card"
      onClick={onClick}
      style={{
        padding: 20,
        cursor: onClick ? 'pointer' : 'default',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}
      id={`prediction-card-${prediction.id}`}
    >
      {/* Status + Date */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className={cfg.className}>
          {cfg.icon}
          {cfg.label}
        </span>
        <span style={{ fontSize: 11, color: 'var(--color-text-faint)' }}>
          {formatDate(created_at)}
        </span>
      </div>

      {/* Breed */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {speciesName === 'Cat' ? (
            <Cat size={16} color="var(--color-accent)" />
          ) : speciesName === 'Dog' ? (
            <Dog size={16} color="#6366f1" />
          ) : null}
          <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-text)' }}>
            {breedName}
          </span>
        </div>
        {speciesName && (
          <span style={{ fontSize: 12, color: 'var(--color-text-muted)', marginLeft: 24 }}>
            {speciesName}
          </span>
        )}
      </div>

      {/* Score bar */}
      {score !== null && score !== undefined && status === 'COMPLETED' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ fontSize: 11, color: 'var(--color-text-faint)' }}>Score</span>
            <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-accent)' }}>
              {Math.round(score * 100)}%
            </span>
          </div>
          <div style={{
            height: 4,
            background: 'rgba(139,92,246,0.15)',
            borderRadius: 2,
            overflow: 'hidden',
          }}>
            <div style={{
              height: '100%',
              width: `${score * 100}%`,
              background: 'var(--gradient-brand)',
              borderRadius: 2,
              transition: 'width 0.8s ease',
            }} />
          </div>
        </div>
      )}

      {/* Image key (truncated) */}
      <div style={{
        fontSize: 10,
        color: 'var(--color-text-faint)',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        borderTop: '1px solid var(--color-border)',
        paddingTop: 8,
      }}>
        🗂 {prediction.image_key}
      </div>
    </div>
  );
}
