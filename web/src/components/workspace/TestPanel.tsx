'use client';

import * as Tabs from '@radix-ui/react-tabs';
import { useLocale } from '@/context/LocaleContext';
import { useProblemStore } from '@/store/problemStore';
import { TestCasesView } from './TestCasesView';
import { TestResultsView } from './TestResultsView';
import { CaseBlock } from './CaseBlock';
import type { AlgoCase, CodeMode, Test } from '@/lib/types';

interface TestPanelProps {
  tests: Test[];
  functionName: string;
  /** Hot 100 cases for the active answer mode */
  cases?: AlgoCase[];
  codeMode?: CodeMode;
}

export function TestPanel({ tests, functionName, cases, codeMode = 'core' }: TestPanelProps) {
  const { t } = useLocale();
  const bottomTab = useProblemStore((s) => s.bottomTab);
  const setBottomTab = useProblemStore((s) => s.setBottomTab);
  const runResult = useProblemStore((s) => s.runResult);

  return (
    <div className="flex flex-col overflow-hidden flex-[2] min-h-0" style={{ borderTop: '1px solid var(--line)', background: 'var(--bg)' }}>
      <Tabs.Root
        value={bottomTab}
        onValueChange={(v) => setBottomTab(v as 'testcases' | 'testresults')}
        className="flex flex-col h-full"
      >
        <Tabs.List className="flex px-4 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
          {(['testcases', 'testresults'] as const).map((tab) => (
            <Tabs.Trigger
              key={tab}
              value={tab}
              className="px-3 py-2 text-sm text-text-2 transition-colors -mb-px data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:hover:text-text"
            >
              {t(tab === 'testcases' ? 'testCasesTab' : 'testResultsTab')}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        <Tabs.Content value="testcases" className="flex-1 overflow-hidden">
          {cases ? (
            <AlgoCasesView cases={cases} codeMode={codeMode} />
          ) : (
            <TestCasesView tests={tests} functionName={functionName} />
          )}
        </Tabs.Content>
        <Tabs.Content value="testresults" className="flex-1 overflow-hidden">
          <TestResultsView result={runResult} tests={tests} functionName={functionName} algoMode={cases ? codeMode : undefined} />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}

/** Hot 100 samples: the two visible cases that Run executes */
function AlgoCasesView({ cases, codeMode }: { cases: AlgoCase[]; codeMode: CodeMode }) {
  const { t } = useLocale();
  const selectedCaseIndex = useProblemStore((s) => s.selectedCaseIndex);
  const setSelectedCaseIndex = useProblemStore((s) => s.setSelectedCaseIndex);
  const samples = cases.slice(0, 2);
  const active = samples[Math.min(selectedCaseIndex, samples.length - 1)];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-center gap-1 px-4 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
        {samples.map((_, i) => (
          <button
            key={i}
            onClick={() => setSelectedCaseIndex(i)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${active === samples[i] ? 'text-accent' : 'text-text-2 hover:text-text'}`}
            style={active === samples[i] ? { background: 'var(--accent-wash)' } : undefined}
          >
            Case {i + 1}
          </button>
        ))}
      </div>
      {active && (
        <div className="flex-1 overflow-auto px-4 py-3 space-y-3">
          <CaseBlock label={codeMode === 'acm' ? t('acmStdin') : t('caseInput')} text={active.input ?? ''} />
          <CaseBlock label={codeMode === 'acm' ? t('acmExpected') : t('caseExpected')} text={active.expected ?? ''} />
        </div>
      )}
    </div>
  );
}
