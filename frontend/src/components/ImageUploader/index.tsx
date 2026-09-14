import { useState, useRef, useCallback } from 'react';
import { Upload, ImageIcon, X } from 'lucide-react';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
  isUploading: boolean;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_MB = 10;

export function ImageUploader({ onFileSelected, isUploading }: ImageUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSet = useCallback((file: File) => {
    setValidationError(null);

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setValidationError('Chỉ hỗ trợ JPG, PNG, WebP');
      return;
    }

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setValidationError(`File quá lớn (tối đa ${MAX_SIZE_MB}MB)`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreview(url);
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
              {isDragOver ? 'Thả ảnh vào đây!' : 'Kéo & thả ảnh vào đây'}
            </h3>

            <p style={{ color: 'var(--color-text-muted)', fontSize: 14, marginBottom: 20 }}>
              hoặc <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>click để chọn file</span>
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
              }}
            />

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
                title="Xóa ảnh"
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

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
            <button
              className="btn btn-ghost"
              onClick={handleClear}
              style={{ flex: 1 }}
              id="change-image-btn"
            >
              <Upload size={16} />
              Đổi ảnh
            </button>
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isUploading}
              style={{ flex: 2 }}
              id="classify-btn"
            >
              {isUploading ? (
                <>
                  <span className="animate-spin" style={{
                    width: 16,
                    height: 16,
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderTopColor: 'white',
                    borderRadius: '50%',
                    display: 'inline-block',
                  }} />
                  Đang phân tích...
                </>
              ) : (
                <>
                  <ImageIcon size={16} />
                  Phân loại ngay
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
