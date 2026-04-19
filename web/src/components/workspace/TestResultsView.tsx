'use client';

import { CheckCircle, XCircle } from 'lucide-react';
import { PythonCode } from '@/lib/pythonHighlight';
import { useLocale } from '@/context/LocaleContext';
import { useProblemStore } from '@/store/problemStore';
import type { SubmissionResult, Test } from '@/lib/types';

interface TestResultsViewProps {
  result: SubmissionResult | null;
  tests: Test[];
  functionName: string;
}

function formatTestCode(code: string, functionName: string): string {
  return code
    .replace(/\{fn\}/g, functionName)
    .split('\n')
    .filter(l => !l.startsWith('import ') && !l.startsWith('from '))
    .join('\n')
    .trim();
}

export function TestResultsView({ result, tests, functionName }: TestResultsViewProps) {
  const { t } = useLocale();
  const { selectedCaseIndex, setSelectedCaseIndex } = useProblemStore();

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-text-3">
        {t('runToSeeResults')}
      </div>
    );
  }

  if (result.error && result.results.length === 0) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
          <span className="text-sm font-medium text-hard">{t('failed')}</span>
        </div>
        <div className="flex-1 overflow-auto px-4 py-3">
          <pre className="p-3 rounded-lg text-hard text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words" style={{ background: 'color-mix(in oklab, var(--hard) 5%, var(--bg-elev))' }}>
            {result.error}
          </pre>
        </div>
      </div>
    );
  }

  const activeIndex = Math.min(selectedCaseIndex, result.results.length - 1);
  const activeResult = result.results[activeIndex];
  const activeTest = activeIndex < tests.length ? tests[activeIndex] : null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
        <span className={`text-sm font-medium ${result.allPassed ? 'text-easy' : 'text-hard'}`}>
          {result.allPassed ? t('allPassed') : t('passedCount', { passed: result.passed, total: result.total })}
        </span>
        <span className="text-text-3 text-xs ml-auto">{result.totalTimeMs.toFixed(0)}ms</span>
      </div>

      <div className="flex items-center gap-1 px-4 py-2 flex-shrink-0 overflow-x-auto" style={{ borderBottom: '1px solid var(--line)' }}>
        {result.results.map((r, i) => (
          <button
            key={i}
            onClick={() => setSelectedCaseIndex(i)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
              i === activeIndex ? 'text-accent' : 'text-text-2 hover:text-text'
            }`}
            style={i === activeIndex ? { background: 'var(--accent-wash)' } : undefined}
          >
            {r.passed
              ? <CheckCircle className="w-3 h-3 text-easy" />
              : <XCircle className="w-3 h-3 text-hard" />}
            {r.name}
          </button>
        ))}
      </div>

      {activeResult && (
        <div className="flex-1 overflow-auto px-4 py-3 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className={`font-medium ${activeResult.passed ? 'text-easy' : 'text-hard'}`}>
              {activeResult.passed ? t('passed') : t('failed')}
            </span>
            <span className="text-text-3">{activeResult.execTimeMs.toFixed(1)}ms</span>
          </div>

          {activeTest && !activeResult.passed && (
            <div>
              <div className="text-xs text-text-3 mb-1">{t('testCasesTab')}</div>
              <pre className="p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed" style={{ background: 'var(--bg-sunken)' }}>
                <PythonCode code={formatTestCode(activeTest.code, functionName)} />
              </pre>
            </div>
          )}

          {activeResult.error && (
            <div>
              <div className="text-xs text-text-3 mb-1">Error</div>
              <pre className="p-3 rounded-lg text-hard text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words" style={{ background: 'color-mix(in oklab, var(--hard) 5%, var(--bg-elev))' }}>
                {activeResult.error}
              </pre>
            </div>
          )}

          {activeResult.output && (
            <div>
              <div className="text-xs text-text-3 mb-1">Output</div>
              <pre className="p-3 rounded-lg text-text text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words" style={{ background: 'var(--bg-sunken)' }}>
                {activeResult.output}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
