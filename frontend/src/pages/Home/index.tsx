import { useState, useCallback } from 'react';
import { Cat, Zap, Sparkles } from 'lucide-react';
import { ImageUploader } from '../../components/ImageUploader';
import { PredictionResult } from '../../components/PredictionResult';
import { predictionApi } from '../../services/predictionApi';
import { usePrediction } from '../../hooks/usePrediction';

export function HomePage() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const { prediction, isPolling, error: pollError, startPolling, reset } = usePrediction();

  const handleFileSelected = useCallback(async (file: File) => {
    setUploadError(null);
    setIsUploading(true);

    // Create local preview URL immediately
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    try {
      const created = await predictionApi.createPrediction(file);
      startPolling(created.id);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload thất bại';
      setUploadError(message);
    } finally {
      setIsUploading(false);
    }
  }, [startPolling]);

  const handleReset = useCallback(() => {
    reset();
    setUploadError(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [reset, previewUrl]);

  const showResult = prediction !== null;

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      padding: '0 24px 60px',
      maxWidth: 1100,
      margin: '0 auto',
      width: '100%',
    }}>

      {/* Hero Header */}
      <header style={{ textAlign: 'center', padding: '60px 0 48px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 16px',
          background: 'rgba(139,92,246,0.1)',
          border: '1px solid rgba(139,92,246,0.25)',
          borderRadius: 999,
          fontSize: 13,
          color: 'var(--color-accent)',
          fontWeight: 600,
          marginBottom: 24,
        }}>
          <Sparkles size={14} />
          CLIP Vision Model — Oxford-IIIT Pet Dataset
        </div>

        <h1 style={{
          fontSize: 'clamp(36px, 5vw, 64px)',
          fontWeight: 900,
          letterSpacing: '-0.03em',
          lineHeight: 1.05,
          marginBottom: 20,
        }}>
          <span className="gradient-text">Pet Breed</span>{' '}
          <span style={{ color: 'var(--color-text)' }}>Classifier</span>
        </h1>

        <p style={{
          color: 'var(--color-text-muted)',
          fontSize: 18,
          maxWidth: 520,
          margin: '0 auto',
          lineHeight: 1.7,
        }}>
          Upload ảnh thú cưng và AI sẽ nhận diện{' '}
          <strong style={{ color: 'var(--color-text)' }}>37 giống chó/mèo</strong>{' '}
          từ bộ dataset Oxford-IIIT Pet
        </p>

        {/* Stats row */}
        <div style={{
          display: 'flex',
          gap: 32,
          justifyContent: 'center',
          marginTop: 32,
          flexWrap: 'wrap',
        }}>
          {[
            { label: '37 Breeds', sub: 'Chó & mèo', icon: <Cat size={16} /> },
            { label: '80.16%', sub: 'Accuracy Zero-Shot', icon: <Zap size={16} /> },
            { label: 'Top-K', sub: 'Kết quả chi tiết', icon: <Sparkles size={16} /> },
          ].map((stat) => (
            <div key={stat.label} style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 4,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ color: 'var(--color-accent)' }}>{stat.icon}</span>
                <span style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)' }}>{stat.label}</span>
              </div>
              <span style={{ fontSize: 12, color: 'var(--color-text-faint)' }}>{stat.sub}</span>
            </div>
          ))}
        </div>
      </header>

      {/* Main content */}
      {!showResult ? (
        /* Upload area */
        <div style={{ maxWidth: 680, margin: '0 auto', width: '100%' }}>
          <ImageUploader
            onFileSelected={handleFileSelected}
            isUploading={isUploading || isPolling}
          />

          {uploadError && (
            <div style={{
              marginTop: 16,
              padding: '12px 16px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 10,
              color: 'var(--color-error)',
              fontSize: 14,
            }}>
              ⚠ {uploadError}
            </div>
          )}

          {pollError && (
            <div style={{
              marginTop: 16,
              padding: '12px 16px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 10,
              color: 'var(--color-error)',
              fontSize: 14,
            }}>
              ⚠ {pollError}
            </div>
          )}
        </div>
      ) : (
        /* Result view — 2 column on wide screens */
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: 24,
          alignItems: 'start',
          maxWidth: 900,
          margin: '0 auto',
          width: '100%',
        }}>
          {/* Left: image preview or uploader */}
          <div>
            <h3 style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--color-text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              marginBottom: 16,
            }}>
              Ảnh đã upload
            </h3>

            {previewUrl ? (
              <div style={{
                borderRadius: 16,
                overflow: 'hidden',
                border: '1px solid rgba(139,92,246,0.25)',
                background: 'var(--color-surface)',
                aspectRatio: '1 / 1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <img
                  src={previewUrl}
                  alt="Ảnh đã upload"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block',
                  }}
                />
              </div>
            ) : (
              <ImageUploader
                onFileSelected={handleFileSelected}
                isUploading={isUploading || isPolling}
              />
            )}

            <button
              className="btn btn-ghost"
              onClick={handleReset}
              style={{ marginTop: 12, width: '100%' }}
              id="new-prediction-btn"
            >
              Phân loại ảnh mới
            </button>
          </div>

          {/* Right: result */}
          <div>
            <h3 style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--color-text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              marginBottom: 16,
            }}>
              Kết quả phân loại
            </h3>
            <PredictionResult prediction={prediction} />
          </div>
        </div>
      )}
    </div>
  );
}
