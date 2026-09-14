import { useState, useRef, useCallback } from 'react';
import { Upload, ImageIcon, X, Loader } from 'lucide-react';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
  isUploading: boolean;
  children?: React.ReactNode;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_MB = 10;

export function ImageUploader({ onFileSelected, isUploading, children }: ImageUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const actionRowRef = useRef<HTMLDivElement>(null);

  const validateAndSet = useCallback((file: File) => {
    setValidationError(null);

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setValidationError('Only JPG, PNG, WebP are supported');
      return;
    }

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setValidationError(`File is too large (max ${MAX_SIZE_MB}MB)`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
    
    // Auto-scroll slightly down to make room for dropdown but not too deep
    setTimeout(() => {
      if (actionRowRef.current) {
        const elementRect = actionRowRef.current.getBoundingClientRect();
        // Scroll so the action row is about 150px from the bottom of the screen
        // This ensures the image stays in view while leaving space for the dropdown
        const offset = elementRect.top + window.scrollY - (window.innerHeight - 200);
        window.scrollTo({ top: offset, behavior: 'smooth' });
      }
    }, 100);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) validateAndSet(file);
  }, [validateAndSet]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) validateAndSet(file);
  }, [validateAndSet]);

  const handleClear = useCallback(() => {
    setPreview(null);
    setSelectedFile(null);
    setValidationError(null);
    if (inputRef.current) inputRef.current.value = '';
  }, []);

  const handleSubmit = useCallback(() => {
    if (selectedFile) onFileSelected(selectedFile);
  }, [selectedFile, onFileSelected]);

  return (
    <div style={{ width: '100%' }}>
      {/* Drop Zone */}
      {!preview ? (
        <div
          className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          id="upload-dropzone"
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(',')}
            onChange={handleChange}
            style={{ display: 'none' }}
            id="upload-input"
          />

          <div style={{ position: 'relative', zIndex: 1 }}>
            <div
              className="animate-float"
              style={{
                width: 72,
                height: 72,
                borderRadius: '50%',
                background: 'rgba(139,92,246,0.15)',
                border: '1px solid rgba(139,92,246,0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 24px',
              }}
            >
              <Upload size={32} color="var(--color-accent)" />
            </div>

            <h3 style={{
              fontSize: 20,
              fontWeight: 600,
              color: 'var(--color-text)',
              marginBottom: 8,
            }}>
              {isDragOver ? 'Drop image here!' : 'Drag & drop image here'}
            </h3>

            <p style={{ color: 'var(--color-text-muted)', fontSize: 14, marginBottom: 20 }}>
              or <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>click to select file</span>
            </p>

            <div style={{
              display: 'flex',
              gap: 8,
              justifyContent: 'center',
              flexWrap: 'wrap',
            }}>
              {['JPG', 'PNG', 'WebP'].map((fmt) => (
                <span key={fmt} style={{
                  padding: '4px 10px',
                  background: 'rgba(139,92,246,0.1)',
                  border: '1px solid rgba(139,92,246,0.2)',
                  borderRadius: 6,
                  fontSize: 12,
                  color: 'var(--color-text-muted)',
                  fontWeight: 500,
                }}>
                  {fmt}
                </span>
              ))}
              <span style={{
                padding: '4px 10px',
                background: 'rgba(139,92,246,0.1)',
                border: '1px solid rgba(139,92,246,0.2)',
                borderRadius: 6,
                fontSize: 12,
                color: 'var(--color-text-muted)',
                fontWeight: 500,
              }}>
                Max {MAX_SIZE_MB}MB
              </span>
            </div>

            {validationError && (
              <p style={{ color: 'var(--color-error)', fontSize: 13, marginTop: 16, fontWeight: 500 }}>
                ⚠ {validationError}
              </p>
            )}
          </div>
        </div>
      ) : (
        /* Image Preview */
        <div className="animate-fade-in" style={{ width: '100%' }}>
          <div style={{
            position: 'relative',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
            border: '1px solid var(--color-border-strong)',
            background: 'var(--color-surface)',
          }}>
            <img
              src={preview}
              alt="Preview"
              style={{
                width: '100%',
                maxHeight: 360,
                objectFit: 'contain',
                display: 'block',
                opacity: isUploading ? 0.7 : 1,
                transition: 'opacity 0.3s'
              }}
            />
            {isUploading && (
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

            {/* Overlay controls */}
            <div style={{
              position: 'absolute',
              top: 12,
              right: 12,
              display: 'flex',
              gap: 8,
            }}>
              <button
                onClick={handleClear}
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: '50%',
                  background: 'rgba(0,0,0,0.7)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  backdropFilter: 'blur(10px)',
                }}
                title="Remove image"
                id="clear-image-btn"
              >
                <X size={16} color="white" />
              </button>
            </div>

            {/* File info bar */}
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              padding: '12px 16px',
              background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%)',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <ImageIcon size={14} color="rgba(255,255,255,0.7)" />
              <span style={{ fontSize: 13, color: 'rgba(255,255,255,0.8)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {selectedFile?.name}
              </span>
              <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>
                {selectedFile ? `${(selectedFile.size / 1024).toFixed(0)} KB` : ''}
              </span>
            </div>
          </div>

          {/* Action Row */}
          <div ref={actionRowRef} style={{ display: 'flex', gap: 12, marginTop: 16, alignItems: 'stretch' }}>
            <div style={{ flex: '0 0 260px', minWidth: 0 }}>
              {children}
            </div>
            
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isUploading}
              style={{ flex: 1, height: '100%', minHeight: '44px' }}
              id="classify-btn"
            >
              {isUploading ? (
                <>
                  <Loader size={18} className="animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <ImageIcon size={18} />
                  Classify Now
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
