import React, { useEffect, useState } from 'react';
import type { ModelInfo, ModelMethod } from '../../types/prediction';
import { predictionApi } from '../../services/predictionApi';
import { ChevronDown, Cpu, Sparkles, ImageIcon, TextCursorInput } from 'lucide-react';

interface ConfigPanelProps {
  selectedMethod: ModelMethod;
  onSelectMethod: (method: ModelMethod) => void;
  customPrompt: string;
  onPromptChange: (prompt: string) => void;
  onClassify: () => void;
}

const DEFAULT_PROMPT = "Picture of a pet with breed: {raw_name}, which is a type of animal.";

const getModelDisplayName = (m: ModelInfo | {id: string, name: string}) => {
  let name = m.name.replace(/\s*\(.*?\)\s*/g, '').trim();
  if (m.id === 'coop') return `${name} (Recommended 🌟)`;
  if (m.id === 'zero-shot') return `${name} (Fastest ⚡)`;
  return name;
};

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  selectedMethod,
  onSelectMethod,
  customPrompt,
  onPromptChange,
  onClassify
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  
  useEffect(() => {
    predictionApi.listModels()
      .then(data => setModels(data))
      .catch(err => console.error("Failed to fetch models", err));
  }, []);

  return (
    <div style={{
      background: 'var(--color-surface)',
      borderRadius: 24,
      border: '1px solid var(--color-border)',
      boxShadow: '0 10px 40px -10px rgba(0,0,0,0.05)',
      height: '100%',
      minHeight: 400,
      display: 'flex',
      flexDirection: 'column',
      padding: 32,
    }}>
      <h2 style={{
        fontSize: 20,
        fontWeight: 700,
        color: 'var(--color-text)',
        marginBottom: 24,
        letterSpacing: '-0.01em',
        display: 'flex',
        alignItems: 'center',
        gap: 8
      }}>
        <Sparkles size={20} className="text-indigo-500" />
        AI Configuration
      </h2>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 24 }}>
        {/* Model Backbone */}
        <div>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Model Backbone
          </label>
          <div style={{ display: 'flex', gap: 12 }}>
            <button 
              className="flex-1 px-4 py-3 bg-indigo-50 dark:bg-indigo-900/20 border-2 border-indigo-500 rounded-xl text-indigo-700 dark:text-indigo-300 font-semibold text-sm flex items-center justify-center gap-2 transition-all"
            >
              <Cpu size={16} /> CLIP
            </button>
            <button 
              disabled
              className="flex-1 px-4 py-3 bg-slate-50 dark:bg-slate-800/50 border-2 border-transparent rounded-xl text-slate-400 font-semibold text-sm flex items-center justify-center gap-2 cursor-not-allowed opacity-60"
              title="Coming soon"
            >
              <Cpu size={16} /> ResNet
            </button>
          </div>
        </div>

        {/* Adaptive Method */}
        <div>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Adaptive Method
          </label>
          <div className="relative">
            <select 
              value={selectedMethod}
              onChange={(e) => onSelectMethod(e.target.value as ModelMethod)}
              style={{
                width: '100%',
                appearance: 'none',
                padding: '14px 16px',
                background: 'var(--color-bg)',
                border: '1px solid var(--color-border-strong)',
                borderRadius: 12,
                fontSize: 15,
                fontWeight: 500,
                color: 'var(--color-text)',
                cursor: 'pointer',
                outline: 'none',
              }}
              className="hover:border-indigo-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition-all"
            >
              {models.length > 0 ? (
                models.map(m => (
                  <option key={m.id} value={m.id}>
                    {getModelDisplayName(m)}
                  </option>
                ))
              ) : (
                <>
                  <option value="zero-shot">{getModelDisplayName({id: 'zero-shot', name: 'Zero-Shot'})}</option>
                  <option value="tip-adapter-f">Tip-Adapter-F</option>
                  <option value="clip-adapter">CLIP-Adapter</option>
                  <option value="coop">{getModelDisplayName({id: 'coop', name: 'CoOp'})}</option>
                </>
              )}
            </select>
            <ChevronDown size={18} style={{ position: 'absolute', right: 16, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--color-text-muted)' }} />
          </div>
        </div>

        {/* Custom Prompt (Only for CoOp) */}
        {selectedMethod === 'coop' && (
          <div className="animate-fade-in">
            <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Custom Prompt Context
            </label>
            <div className="relative">
              <TextCursorInput size={16} className="absolute left-3 top-3.5 text-slate-400" />
              <input 
                type="text"
                value={customPrompt}
                onChange={(e) => onPromptChange(e.target.value)}
                placeholder={DEFAULT_PROMPT}
                style={{
                  width: '100%',
                  padding: '12px 16px 12px 36px',
                  background: 'var(--color-bg)',
                  border: '1px solid var(--color-border-strong)',
                  borderRadius: 12,
                  fontSize: 14,
                  color: 'var(--color-text)',
                  outline: 'none',
                }}
                className="hover:border-indigo-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition-all placeholder-slate-400"
              />
            </div>
            <p style={{ fontSize: 12, color: 'var(--color-text-faint)', marginTop: 6 }}>
              You can optionally enter a custom prompt template. Leave blank to use the default optimal template.
            </p>
          </div>
        )}
      </div>

      <button
        onClick={onClassify}
        style={{
          width: '100%',
          padding: '16px',
          background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
          color: 'white',
          border: 'none',
          borderRadius: 14,
          fontSize: 16,
          fontWeight: 700,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          boxShadow: '0 4px 14px rgba(99,102,241,0.3)',
          marginTop: 24,
          transition: 'all 0.2s ease',
        }}
        className="hover:shadow-[0_6px_20px_rgba(99,102,241,0.4)] hover:-translate-y-0.5 active:translate-y-0 active:shadow-[0_2px_8px_rgba(99,102,241,0.3)]"
      >
        <ImageIcon size={20} />
        Classify Now
      </button>
    </div>
  );
};
