import { useState, useRef, useCallback } from 'react';
import { Upload } from 'lucide-react';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_MB = 10;

export function ImageUploader({ onFileSelected }: ImageUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

    onFileSelected(file);
  }, [onFileSelected]);

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

  return (
    <div style={{ width: '100%' }}>
      <div
        className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{ cursor: 'pointer', padding: '60px 20px' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleChange}
          style={{ display: 'none' }}
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
    </div>
  );
}
