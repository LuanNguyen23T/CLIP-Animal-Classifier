import { useState, useCallback } from 'react';
import { Cat, Zap, Sparkles } from 'lucide-react';
import { ImageUploader } from '../../components/ImageUploader';
import { PredictionResult } from '../../components/PredictionResult';
import ModelSelector from '../../components/ModelSelector';
import { predictionApi } from '../../services/predictionApi';
import { usePrediction } from '../../hooks/usePrediction';
import type { ModelMethod } from '../../types/prediction';

export function HomePage() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<ModelMethod>('zero-shot');
  const { prediction, isPolling, error: pollError, startPolling, reset } = usePrediction();

  const handleFileSelected = useCallback(async (file: File) => {
    setUploadError(null);
    setIsUploading(true);

    // Create local preview URL immediately
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    try {
      const created = await predictionApi.createPrediction(file, selectedMethod);
      startPolling(created.id);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(message);
    } finally {
      setIsUploading(false);
    }
  }, [startPolling, selectedMethod]);

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
          AI-Powered Pet Identification
        </div>

        <h1 style={{
          fontSize: 'clamp(36px, 5vw, 64px)',
          fontWeight: 900,
          letterSpacing: '-0.03em',
          lineHeight: 1.05,
          marginBottom: 20,
        }}>
          <span className="gradient-text">Discover</span>{' '}
          <span style={{ color: 'var(--color-text)' }}>Your Pet's Breed</span>
        </h1>

        <p style={{
          color: 'var(--color-text-muted)',
          fontSize: 18,
          maxWidth: 560,
          margin: '0 auto',
          lineHeight: 1.7,
        }}>
          Upload a photo of your pet and let our AI system quickly and accurately identify dozens of popular dog and cat breeds.
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
            { label: 'Lightning Fast', sub: 'Instant Recognition', icon: <Zap size={16} /> },
            { label: 'Highly Accurate', sub: 'Top-K Analysis', icon: <Sparkles size={16} /> },
            { label: 'Many Breeds', sub: 'Dogs & Cats', icon: <Cat size={16} /> },
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
          >
            <ModelSelector selectedMethod={selectedMethod} onSelectMethod={setSelectedMethod} />
          </ImageUploader>

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
              Uploaded Image
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
                  alt="Uploaded Image"
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
              Classify New Image
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
              Classification Result
            </h3>
            <PredictionResult prediction={prediction} />
          </div>
        </div>
      )}

      {/* Intro / Footer Section */}
      <footer style={{
        marginTop: '80px',
        padding: '40px 0',
        borderTop: '1px solid var(--color-border)',
        textAlign: 'center',
      }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: 'var(--color-text)', marginBottom: 16 }}>
          How It Works
        </h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: 24,
          maxWidth: 900,
          margin: '0 auto',
          textAlign: 'left',
        }}>
          <div style={{ background: 'var(--color-surface)', padding: 24, borderRadius: 16, border: '1px solid var(--color-border)', boxShadow: '0 4px 20px rgba(0,0,0,0.02)' }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-text)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 24, height: 24, background: 'var(--color-accent-glow)', color: 'var(--color-accent)', borderRadius: '50%', fontSize: 12 }}>1</span>
              Upload Image
            </h3>
            <p style={{ fontSize: 14, color: 'var(--color-text-muted)', lineHeight: 1.6 }}>Choose a clear photo of a dog or cat. Our system works best with well-lit, front-facing images.</p>
          </div>
          <div style={{ background: 'var(--color-surface)', padding: 24, borderRadius: 16, border: '1px solid var(--color-border)', boxShadow: '0 4px 20px rgba(0,0,0,0.02)' }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-text)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 24, height: 24, background: 'var(--color-accent-glow)', color: 'var(--color-accent)', borderRadius: '50%', fontSize: 12 }}>2</span>
              Select AI Model
            </h3>
            <p style={{ fontSize: 14, color: 'var(--color-text-muted)', lineHeight: 1.6 }}>Choose from our advanced CLIP models. CoOp provides highest accuracy, while Zero-Shot is lightning fast.</p>
          </div>
          <div style={{ background: 'var(--color-surface)', padding: 24, borderRadius: 16, border: '1px solid var(--color-border)', boxShadow: '0 4px 20px rgba(0,0,0,0.02)' }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-text)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 24, height: 24, background: 'var(--color-accent-glow)', color: 'var(--color-accent)', borderRadius: '50%', fontSize: 12 }}>3</span>
              Get Results
            </h3>
            <p style={{ fontSize: 14, color: 'var(--color-text-muted)', lineHeight: 1.6 }}>Our neural networks will analyze the image in milliseconds and return the top 5 most likely breeds.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
