import React, { useEffect, useState, useRef } from 'react';
import type { ModelInfo, ModelMethod } from '../../types/prediction';
import { predictionApi } from '../../services/predictionApi';
import { ChevronDown, Check, Zap, Star, Cpu, Layers } from 'lucide-react';

interface ModelSelectorProps {
  selectedMethod: ModelMethod;
  onSelectMethod: (method: ModelMethod) => void;
}

const getModelDisplayInfo = (model: ModelInfo) => {
  let title = model.name.replace(/\s*\(.*?\)\s*/g, '').trim();
  
  if (model.id === 'zero-shot') title = 'Zero-Shot';
  if (model.id === 'tip-adapter-f') title = 'Tip-Adapter-F';
  if (model.id === 'clip-adapter') title = 'CLIP-Adapter';
  if (model.id === 'coop') title = 'CoOp';

  let badge = null;
  let icon = <Cpu size={18} className="text-slate-500 dark:text-slate-400" />;
  let colorTheme = 'slate';

  if (model.id === 'coop') {
    badge = 'Recommended';
    icon = <Star size={18} className="text-amber-500 dark:text-amber-400" />;
    colorTheme = 'amber';
  } else if (model.id === 'zero-shot') {
    badge = 'Fastest';
    icon = <Zap size={18} className="text-blue-500 dark:text-blue-400" />;
    colorTheme = 'blue';
  } else if (model.id === 'clip-adapter') {
    icon = <Layers size={18} className="text-slate-500 dark:text-slate-400" />;
  }

  // model.description is provided by backend
  return { title, description: model.description, badge, icon, colorTheme };
};

const ModelSelector: React.FC<ModelSelectorProps> = ({ selectedMethod, onSelectMethod }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    predictionApi.listModels()
      .then(data => {
        setModels(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch models", err);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (loading) {
    return <div className="text-center p-4 text-sm text-slate-500">Loading AI models...</div>;
  }

  const selectedModel = models.find(m => m.id === selectedMethod) || models[0];
  const selectedInfo = selectedModel ? getModelDisplayInfo(selectedModel) : null;

  return (
    <div className="flex flex-col items-center relative h-full w-full" ref={dropdownRef}>
      {/* Nút Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          w-full h-full relative group flex items-center justify-between px-3 py-2
          bg-white dark:bg-slate-900 border 
          ${isOpen ? 'border-indigo-500 ring-2 ring-indigo-500/10 dark:ring-indigo-500/20 shadow-md' : 'border-slate-200 dark:border-slate-700 shadow-sm hover:border-slate-300 dark:hover:border-slate-600'} 
          rounded-xl transition-all duration-300 ease-out text-left
        `}
      >
        <div className="flex items-center gap-3 overflow-hidden">
          {/* Icon Box */}
          <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-100 dark:border-slate-700/50 group-hover:scale-105 transition-transform duration-300">
            {selectedInfo?.icon}
          </div>
          
          {/* Text Content */}
          <div className="flex flex-col truncate pr-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-[14px] text-slate-800 dark:text-slate-100 truncate">
                {selectedInfo?.title}
              </span>
              {selectedInfo?.badge && (
                <span className={`
                  flex items-center px-1.5 py-0.5 rounded-md text-[10px] font-bold tracking-wide uppercase flex-shrink-0
                  ${selectedInfo.colorTheme === 'amber' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400' : ''}
                  ${selectedInfo.colorTheme === 'blue' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400' : ''}
                `}>
                  {selectedInfo.badge}
                </span>
              )}
            </div>
          </div>
        </div>
        
        {/* Chevron Icon */}
        <div className="flex-shrink-0 pl-2">
          <ChevronDown 
            size={18} 
            strokeWidth={2.5}
            className={`text-slate-400 transition-transform duration-300 ease-in-out ${isOpen ? 'rotate-180 text-indigo-500' : ''}`} 
          />
        </div>
      </button>

      <style>{`
        @keyframes popIn {
          0% { opacity: 0; transform: translateY(-8px) scale(0.98); }
          100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .animate-pop-in {
          animation: popIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .thin-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .thin-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .thin-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.5);
          border-radius: 4px;
        }
      `}</style>

      {isOpen && (
        <div className="absolute top-[calc(100%+8px)] w-full bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border border-slate-200 dark:border-slate-800 rounded-2xl shadow-[0_10px_40px_-10px_rgba(0,0,0,0.1)] dark:shadow-[0_10px_40px_-10px_rgba(0,0,0,0.5)] overflow-hidden z-[100] animate-pop-in p-1.5">
          <div className="max-h-[300px] overflow-y-auto rounded-xl thin-scrollbar">
            {models.map((model) => {
              const info = getModelDisplayInfo(model);
              const isSelected = model.id === selectedMethod;
              
              return (
                <div
                  key={model.id}
                  onClick={() => {
                    onSelectMethod(model.id);
                    setIsOpen(false);
                  }}
                  className={`
                    w-full relative group flex items-center justify-between p-3 mb-1 last:mb-0 rounded-xl cursor-pointer transition-all duration-200
                    ${isSelected 
                      ? 'bg-indigo-50/80 dark:bg-indigo-500/10' 
                      : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }
                  `}
                >
                  <div className="flex items-center gap-3 overflow-hidden">
                    {/* Icon cho từng item */}
                    <div className={`
                      flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-lg transition-colors duration-200
                      ${isSelected ? 'bg-white dark:bg-indigo-500/20 shadow-sm' : 'bg-transparent group-hover:bg-white dark:group-hover:bg-slate-700'}
                    `}>
                      {info.icon}
                    </div>
                    
                    {/* Text cho từng item */}
                    <div className="flex flex-col truncate pr-2">
                      <div className="flex items-center gap-2">
                        <span className={`font-semibold text-[13px] transition-colors duration-200 truncate ${isSelected ? 'text-indigo-700 dark:text-indigo-300' : 'text-slate-700 dark:text-slate-200'}`}>
                          {info.title}
                        </span>
                        {info.badge && (
                          <span className={`
                            flex-shrink-0 px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider uppercase
                            ${info.colorTheme === 'amber' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-400' : ''}
                            ${info.colorTheme === 'blue' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-400' : ''}
                          `}>
                            {info.badge}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Dấu Check */}
                  {isSelected && (
                    <div className="flex-shrink-0 animate-fade-in pl-2">
                      <Check size={18} strokeWidth={3} className="text-indigo-600 dark:text-indigo-400" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
