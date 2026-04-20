'use client';

import { Plus, X } from 'lucide-react';
import { PythonCode } from '@/lib/pythonHighlight';
import { useLocale } from '@/context/LocaleContext';
import { useProblemStore } from '@/store/problemStore';
import { formatTestCode } from '@/lib/problemContext';
import type { Test } from '@/lib/types';

interface TestCasesViewProps {
  tests: Test[];
  functionName: string;
}

export function TestCasesView({ tests, functionName }: TestCasesViewProps) {
  const { t } = useLocale();
  const { selectedCaseIndex, setSelectedCaseIndex, customTests, addCustomTest, removeCustomTest, updateCustomTest } = useProblemStore();

  // Only show first 2 sample tests
  const sampleTests = tests.slice(0, 2);
  const allCases = [
    ...sampleTests.map((t, i) => ({ name: `Case ${i + 1}`, code: t.code, custom: false })),
    ...customTests.map((ct, i) => ({ name: ct.name || `${t('customTest')} ${i + 1}`, code: ct.code, custom: true })),
  ];

  const activeCase = allCases[selectedCaseIndex];
  const customOffset = sampleTests.length;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Case selector */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-border flex-shrink-0 overflow-x-auto">
        {allCases.map((c, i) => (
          <button
            key={i}
            onClick={() => setSelectedCaseIndex(i)}
            className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
              i === selectedCaseIndex
                ? 'bg-accent/10 text-accent'
                : 'text-text-secondary hover:text-text-primary hover:bg-surface-secondary'
            }`}
          >
            {c.name}
            {c.custom && (
              <span
                role="button"
                onClick={(e) => {
                  e.stopPropagation();
                  removeCustomTest(i - customOffset);
                  if (selectedCaseIndex >= allCases.length - 1) setSelectedCaseIndex(Math.max(0, allCases.length - 2));
                }}
                className="ml-1 hover:text-hard"
              >
                <X className="w-2.5 h-2.5" />
              </span>
            )}
          </button>
        ))}
        <button
          onClick={() => {
            addCustomTest({ name: '', code: '' });
            setSelectedCaseIndex(allCases.length);
          }}
          className="p-1 rounded-full text-text-tertiary hover:text-text-primary hover:bg-surface-secondary transition-colors flex-shrink-0"
          title={t('addTestCase')}
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Case content */}
      <div className="flex-1 overflow-auto px-4 py-3">
        {activeCase ? (
          activeCase.custom ? (
            <textarea
              className="w-full h-full min-h-[80px] text-xs font-mono bg-surface-secondary rounded-lg p-3 text-text-primary resize-none focus:outline-none focus:ring-1 focus:ring-accent/50 border border-border"
              placeholder="# 输入自定义测试代码..."
              value={activeCase.code}
              onChange={(e) => updateCustomTest(selectedCaseIndex - customOffset, { name: activeCase.name, code: e.target.value })}
            />
          ) : (
            <pre className="text-xs font-mono bg-surface-secondary rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-words leading-relaxed">
              <PythonCode code={formatTestCode(activeCase.code, functionName)} />
            </pre>
          )
        ) : null}
      </div>
    </div>
  );
}
