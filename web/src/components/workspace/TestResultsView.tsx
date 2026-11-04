'use client';

import { CheckCircle, XCircle, Clock } from 'lucide-react';
import { PythonCode } from '@/lib/pythonHighlight';
import { useLocale } from '@/context/LocaleContext';
import { useProblemStore } from '@/store/problemStore';
import { CaseBlock } from './CaseBlock';
import type { CodeMode, SubmissionResult, Test, TestResult } from '@/lib/types';

interface TestResultsViewProps {
  result: SubmissionResult | null;
  tests: Test[];
  functionName: string;
  /** Set for Hot 100: results carry input / expected / actual strings */
  algoMode?: CodeMode;
}

function formatTestCode(code: string, functionName: string): string {
  return code
    .replace(/\{fn\}/g, functionName)
    .split('\n')
    .filter(l => !l.startsWith('import ') && !l.startsWith('from '))
    .join('\n')
    .trim();
}

function padIndex(i: number): string {
  return String(i + 1).padStart(2, '0');
}

function AlgoResultDetail({ result, mode }: { result: TestResult; mode: CodeMode }) {
  const { t } = useLocale();
  if (result.input == null) {
    return (
      <>
        <p className="text-sm text-text-3">{t('hiddenCase')}</p>
        {result.error && <CaseBlock label="Error" text={result.error} tone="bad" />}
      </>
    );
  }
  const isWrongAnswer = result.error?.startsWith('Wrong Answer');
  return (
    <>
      <CaseBlock label={mode === 'acm' ? t('acmStdin') : t('caseInput')} text={result.input} />
      {result.error && !isWrongAnswer && <CaseBlock label="Error" text={result.error} tone="bad" />}
      {result.actual != null && (
        <CaseBlock
          label={mode === 'acm' ? t('yourStdout') : t('yourAnswer')}
          text={result.actual}
          tone={result.passed ? 'good' : 'bad'}
        />
      )}
      <CaseBlock label={mode === 'acm' ? t('acmExpected') : t('caseExpected')} text={result.expected ?? ''} />
      {isWrongAnswer && result.error !== 'Wrong Answer' && <p className="text-xs text-hard">{result.error}</p>}
      {result.output && <CaseBlock label={t('printOutput')} text={result.output} />}
    </>
  );
}

export function TestResultsView({ result, tests, functionName, algoMode }: TestResultsViewProps) {
  const { t } = useLocale();
  const selectedCaseIndex = useProblemStore((s) => s.selectedCaseIndex);
  const setSelectedCaseIndex = useProblemStore((s) => s.setSelectedCaseIndex);

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
          <XCircle className="w-3.5 h-3.5 text-hard" />
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
      {/* Header: verdict + runtime */}
      <div className="flex items-center gap-3 px-4 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
        {result.allPassed
          ? <CheckCircle className="w-3.5 h-3.5 text-easy" />
          : <XCircle className="w-3.5 h-3.5 text-hard" />}
        <span className={`text-sm font-medium ${result.allPassed ? 'text-easy' : 'text-hard'}`}>
          {result.allPassed ? t('allPassed') : t('passedCount', { passed: result.passed, total: result.total })}
        </span>
        <span className="ml-auto flex items-center gap-1.5 text-text-3 mono text-[11px]">
          <Clock className="w-3 h-3" />
          {result.totalTimeMs.toFixed(0)}ms
        </span>
      </div>

      {/* Two-column: test list | detail */}
      <div className="flex-1 overflow-hidden" style={{ display: 'grid', gridTemplateColumns: '260px 1fr' }}>
        {/* Left: test case list */}
        <div className="overflow-y-auto" style={{ borderRight: '1px solid var(--line)' }}>
          {result.results.map((r, i) => (
            <button
              key={i}
              onClick={() => setSelectedCaseIndex(i)}
              className="w-full text-left flex items-center gap-2.5 cursor-pointer transition-colors"
              style={{
                padding: '8px 14px',
                fontSize: '13px',
                borderLeft: i === activeIndex ? '2px solid var(--accent)' : '2px solid transparent',
                background: i === activeIndex ? 'var(--accent-wash)' : undefined,
              }}
              onMouseEnter={(e) => { if (i !== activeIndex) e.currentTarget.style.background = 'color-mix(in oklab, var(--text) 3%, transparent)'; }}
              onMouseLeave={(e) => { if (i !== activeIndex) e.currentTarget.style.background = ''; }}
            >
              <span
                className="mono text-[11px] w-4 text-center flex-shrink-0"
                style={{ color: r.passed ? 'var(--easy)' : 'var(--hard)' }}
              >
                {r.passed ? '✓' : '✕'}
              </span>
              <span className="mono text-[11px] text-text-3 flex-shrink-0">{padIndex(i)}</span>
              <span className="flex-1 text-text-2 truncate">{r.name}</span>
              <span className="mono text-[11px] text-text-3 flex-shrink-0">{r.execTimeMs.toFixed(1)}ms</span>
            </button>
          ))}
        </div>

        {/* Right: detail pane */}
        {activeResult && (
          <div className="overflow-y-auto px-4 py-3 space-y-3">
            {/* Case header */}
            <div className="flex items-center gap-3">
              <span className="eyebrow">
                CASE {padIndex(activeIndex)} · {activeResult.name}
              </span>
              <span
                className="mono text-[11px] px-1.5 py-0.5 rounded"
                style={{
                  background: activeResult.passed
                    ? 'color-mix(in oklab, var(--easy) 10%, var(--bg-elev))'
                    : 'color-mix(in oklab, var(--hard) 10%, var(--bg-elev))',
                  color: activeResult.passed ? 'var(--easy)' : 'var(--hard)',
                }}
              >
                {activeResult.passed ? 'PASS' : 'FAIL'}
              </span>
              <span className="mono text-[11px] text-text-3">{activeResult.execTimeMs.toFixed(1)}ms</span>
            </div>

            {algoMode && <AlgoResultDetail result={activeResult} mode={algoMode} />}

            {/* Test code */}
            {!algoMode && activeTest && (
              <div>
                <h4 className="eyebrow mb-1.5">{t('testCasesTab')}</h4>
                <pre className="p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed" style={{ background: 'var(--bg-sunken)' }}>
                  <PythonCode code={formatTestCode(activeTest.code, functionName)} />
                </pre>
              </div>
            )}

            {/* Error with expected/got diff */}
            {!algoMode && activeResult.error && (
              <div>
                <h4 className="eyebrow mb-1.5">Error</h4>
                {!activeResult.passed && activeResult.error.includes('Expected') ? (
                  <div className="rounded-lg overflow-hidden text-xs font-mono" style={{ border: '1px solid var(--line)' }}>
                    {activeResult.error.split('\n').map((line, li) => {
                      const isExpected = line.toLowerCase().startsWith('expected');
                      const isGot = line.toLowerCase().startsWith('got') || line.toLowerCase().startsWith('actual');
                      return (
                        <div
                          key={li}
                          className="px-3 py-1.5 whitespace-pre-wrap break-words"
                          style={{
                            background: isExpected
                              ? 'color-mix(in oklab, var(--easy) 6%, var(--bg-elev))'
                              : isGot
                                ? 'color-mix(in oklab, var(--hard) 6%, var(--bg-elev))'
                                : 'var(--bg-elev)',
                            color: isExpected ? 'var(--easy)' : isGot ? 'var(--hard)' : 'var(--text)',
                            borderBottom: '1px solid var(--line)',
                          }}
                        >
                          {line}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <pre className="p-3 rounded-lg text-hard text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words" style={{ background: 'color-mix(in oklab, var(--hard) 5%, var(--bg-elev))' }}>
                    {activeResult.error}
                  </pre>
                )}
              </div>
            )}

            {/* Output */}
            {!algoMode && activeResult.output && (
              <div>
                <h4 className="eyebrow mb-1.5">Output</h4>
                <pre className="p-3 rounded-lg text-text text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words" style={{ background: 'var(--bg-sunken)' }}>
                  {activeResult.output}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
