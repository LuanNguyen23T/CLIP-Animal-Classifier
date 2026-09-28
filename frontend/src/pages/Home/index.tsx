import { useState, useCallback } from 'react';
import { Cat, Zap, Sparkles } from 'lucide-react';
import { ImageUploader } from '../../components/ImageUploader';
import { PredictionResult } from '../../components/PredictionResult';
import { ConfigPanel } from '../../components/ConfigPanel';
import { AnalyzingSpinner } from '../../components/AnalyzingSpinner';
import { predictionApi } from '../../services/predictionApi';
import { usePredictionSession } from '../../hooks/usePredictionSession';
import type { ModelMethod } from '../../types/prediction';

type Step = 'UPLOAD' | 'CONFIG' | 'ANALYZING' | 'RESULT';

export function HomePage() {
  const [step, setStep] = useState<Step>('UPLOAD');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  const [selectedMethod, setSelectedMethod] = useState<ModelMethod>('zero-shot');
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [selectedResultIndex, setSelectedResultIndex] = useState(0);

  const { predictions, activeIds, error: pollError, startPolling, reset } = usePredictionSession();
  const sessionStep: Step | null = activeIds.length > 0
    ? (predictions.length > 0 ? 'RESULT' : 'ANALYZING')
    : null;
  const visibleStep = sessionStep ?? step;
  const visiblePreviewUrl = previewUrl ?? predictions[0]?.image_key ?? null;

  const handleFileSelected = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setSelectedFile(file);
    setStep('CONFIG');
    setUploadError(null);
  }, []);

  const handleClassify = useCallback(async () => {
    if (!selectedFile) return;
    
    setStep('ANALYZING');
    setUploadError(null);
    setSelectedResultIndex(0);

    try {
      if (selectedMethod === 'all') {
        const createdBatch = await predictionApi.createPredictionBatch(selectedFile);
        startPolling(createdBatch.map(p => p.id));
      } else {
        const created = await predictionApi.createPrediction(selectedFile, selectedMethod);
        startPolling([created.id]);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(message);
      setStep('CONFIG'); // Go back to config on error
    }
  }, [selectedFile, selectedMethod, startPolling]);

  const handleReset = useCallback(() => {
    reset();
    setStep('UPLOAD');
    setSelectedFile(null);
    setUploadError(null);
    setSelectedResultIndex(0);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [reset, previewUrl]);

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

      {/* Hero Header - Only show in UPLOAD step */}
      {visibleStep === 'UPLOAD' && (
        <header style={{ textAlign: 'center', padding: '60px 0 48px' }} className="animate-fade-in">
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
      )}

      {/* Main content */}
      <div style={{ marginTop: visibleStep === 'UPLOAD' ? 0 : 40 }} className="animate-fade-in">
        {visibleStep === 'UPLOAD' && (
          <div style={{ maxWidth: 680, margin: '0 auto', width: '100%' }}>
            <ImageUploader onFileSelected={handleFileSelected} />
          </div>
        )}

        {visibleStep !== 'UPLOAD' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
            gap: 32,
            alignItems: 'start',
            maxWidth: 1000,
            margin: '0 auto',
            width: '100%',
          }}>
            {/* Left Column: Image Preview */}
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

              <div style={{
                borderRadius: 24,
                overflow: 'hidden',
                border: '1px solid var(--color-border)',
                background: 'var(--color-surface)',
                boxShadow: '0 10px 40px -10px rgba(0,0,0,0.05)',
                aspectRatio: '1 / 1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}>
                {visiblePreviewUrl && (
                  <img
                    src={visiblePreviewUrl}
                    alt="Uploaded Image"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block',
                      opacity: visibleStep === 'ANALYZING' ? 0.7 : 1,
                      transition: 'opacity 0.3s'
                    }}
                  />
                )}
                
                {/* Scan line effect during analyzing */}
                {visibleStep === 'ANALYZING' && (
                  <>
                    <style>
                      {`
                        @keyframes scan-line {
                          0% { transform: translateY(-100%); opacity: 0; }
                          10% { opacity: 1; }
                          90% { opacity: 1; }
                          100% { transform: translateY(600%); opacity: 0; }
                        }
                      `}
                    </style>
                    <div style={{
                      position: 'absolute',
                      top: 0, left: 0, right: 0,
                      height: '20%',
                      background: 'linear-gradient(to bottom, transparent, rgba(99,102,241,0.2), rgba(99,102,241,0.6))',
                      borderBottom: '3px solid #6366f1',
                      animation: 'scan-line 2s linear infinite',
                      zIndex: 20,
                      pointerEvents: 'none',
                      boxShadow: '0 4px 20px rgba(99,102,241,0.5)'
                    }} />
                  </>
                )}
              </div>

              <button
                className="btn btn-ghost"
                onClick={handleReset}
                style={{ marginTop: 16, width: '100%' }}
              >
                Upload New Image
              </button>
            </div>

            {/* Right Column: Dynamic Panel */}
            <div>
              <h3 style={{
                fontSize: 14,
                fontWeight: 600,
                color: 'var(--color-text-muted)',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                marginBottom: 16,
                visibility: visibleStep === 'RESULT' ? 'visible' : 'hidden' // Only show title for result
              }}>
                {visibleStep === 'RESULT' ? 'Classification Result' : 'Placeholder'}
              </h3>

              {visibleStep === 'CONFIG' && (
                <ConfigPanel 
                  selectedMethod={selectedMethod}
                  onSelectMethod={setSelectedMethod}
                  customPrompt={customPrompt}
                  onPromptChange={setCustomPrompt}
                  onClassify={handleClassify}
                />
              )}

              {visibleStep === 'ANALYZING' && (
                <AnalyzingSpinner />
              )}

              {visibleStep === 'RESULT' && predictions.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {predictions.length > 1 && (
                    <div style={{
                      display: 'flex',
                      gap: 8,
                      overflowX: 'auto',
                      paddingBottom: 8,
                      scrollbarWidth: 'thin'
                    }}>
                      {predictions.map((p, idx) => (
                        <button
                          key={p.id}
                          onClick={() => setSelectedResultIndex(idx)}
                          style={{
                            padding: '6px 12px',
                            borderRadius: 20,
                            border: '1px solid var(--color-border)',
                            background: idx === selectedResultIndex ? 'var(--color-accent)' : 'var(--color-surface)',
                            color: idx === selectedResultIndex ? 'white' : 'var(--color-text)',
                            fontSize: 12,
                            fontWeight: 600,
                            cursor: 'pointer',
                            whiteSpace: 'nowrap',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 6,
                            transition: 'all 0.2s ease',
                          }}
                        >
                          {p.method || 'Unknown'}
                          {p.status === 'COMPLETED' ? ' ✓' : (p.status === 'FAILED' ? ' ✗' : ' ...')}
                        </button>
                      ))}
                    </div>
                  )}
                  {predictions[selectedResultIndex] && (
                    <PredictionResult prediction={predictions[selectedResultIndex]} />
                  )}
                </div>
              )}

              {/* Errors */}
              {(uploadError || pollError) && (
                <div style={{
                  marginTop: 16,
                  padding: '16px',
                  background: 'rgba(239,68,68,0.1)',
                  border: '1px solid rgba(239,68,68,0.3)',
                  borderRadius: 12,
                  color: 'var(--color-error)',
                  fontSize: 14,
                }}>
                  ⚠ {uploadError || pollError}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* Footer - Only show on Home */}
      {visibleStep === 'UPLOAD' && (
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
      )}
    </div>
  );
}
