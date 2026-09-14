import { CheckCircle, XCircle, Clock, Loader2, Award, Cpu, TrendingUp } from 'lucide-react';
import type { PredictionResponse } from '../../types/prediction';
import { PredictionChart } from '../PredictionChart';

interface PredictionResultProps {
  prediction: PredictionResponse;
}

function StatusBadge({ status }: { status: string }) {
  const configs = {
    UPLOADED: { label: 'Đã Upload', className: 'badge badge-uploaded', icon: <Clock size={11} /> },
    PROCESSING: { label: 'Đang xử lý', className: 'badge badge-processing', icon: <Loader2 size={11} className="animate-spin" /> },
    COMPLETED: { label: 'Hoàn thành', className: 'badge badge-completed', icon: <CheckCircle size={11} /> },
    FAILED: { label: 'Thất bại', className: 'badge badge-failed', icon: <XCircle size={11} /> },
  };

  const config = configs[status as keyof typeof configs] ?? configs.UPLOADED;

  return (
    <span className={config.className}>
      {config.icon}
      {config.label}
    </span>
  );
}

function ScoreRing({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (pct / 100) * circumference;
  const color = pct >= 75 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444';

  return (
    <div style={{ position: 'relative', width: 110, height: 110 }}>
      <svg width="110" height="110" style={{ transform: 'rotate(-90deg)' }}>
        {/* Track */}
        <circle cx="55" cy="55" r={radius} fill="none" stroke="rgba(139,92,246,0.15)" strokeWidth="8" />
        {/* Progress */}
        <circle
          cx="55" cy="55" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 1s ease' }}
        />
      </svg>
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <span style={{ fontSize: 22, fontWeight: 800, color, lineHeight: 1 }}>{pct}%</span>
        <span style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>score</span>
      </div>
    </div>
  );
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const { status, predicted_breed, score, top_k_results, model_version, error_message } = prediction;

  /* ---------- PROCESSING ---------- */
  if (status === 'UPLOADED' || status === 'PROCESSING') {
    return (
      <div className="glass-card animate-fade-in" style={{ padding: 40, textAlign: 'center' }}>
        <div style={{
          width: 64,
          height: 64,
          borderRadius: '50%',
          border: '3px solid rgba(139,92,246,0.2)',
          borderTopColor: 'var(--color-accent)',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 20px',
        }} />
        <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Đang phân tích ảnh...</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 14 }}>
          AI đang nhận diện giống thú cưng của bạn
        </p>
      </div>
    );
  }

  /* ---------- FAILED ---------- */
  if (status === 'FAILED') {
    return (
      <div className="glass-card animate-fade-in" style={{
        padding: 32,
        textAlign: 'center',
        borderColor: 'rgba(239,68,68,0.3)',
      }}>
        <XCircle size={48} color="var(--color-error)" style={{ margin: '0 auto 16px' }} />
        <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--color-error)', marginBottom: 8 }}>
          Phân loại thất bại
        </h3>
        {error_message && (
          <p style={{
            color: 'var(--color-text-muted)',
            fontSize: 13,
            background: 'rgba(239,68,68,0.05)',
            padding: '12px 16px',
            borderRadius: 8,
            marginTop: 12,
            textAlign: 'left',
          }}>
            {error_message}
          </p>
        )}
      </div>
    );
  }

  /* ---------- COMPLETED ---------- */
  const breedDisplayName = predicted_breed?.name?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) ?? '—';
  const speciesName = predicted_breed?.species?.name ?? '—';

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Main result card */}
      <div className="glass-card" style={{ padding: 28 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <StatusBadge status={status} />
              {model_version && (
                <span style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  fontSize: 11, color: 'var(--color-text-faint)',
                  padding: '3px 8px',
                  background: 'rgba(139,92,246,0.05)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 6,
                }}>
                  <Cpu size={10} />
                  {model_version}
                </span>
              )}
            </div>
            <h2 style={{ fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
              {breedDisplayName}
            </h2>
            <p style={{ color: 'var(--color-text-muted)', fontSize: 14, marginTop: 4 }}>
              <Award size={13} style={{ display: 'inline', marginRight: 4 }} />
              {speciesName}
            </p>
          </div>

          {/* Score ring */}
          {score !== null && score !== undefined && (
            <ScoreRing score={score} />
          )}
        </div>

        {/* Top-K Chart */}
        {top_k_results && top_k_results.length > 0 && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <TrendingUp size={14} color="var(--color-accent)" />
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-muted)' }}>
                Top {top_k_results.length} kết quả
              </span>
            </div>
            <PredictionChart topKResults={top_k_results} />
          </div>
        )}
      </div>
    </div>
  );
}
