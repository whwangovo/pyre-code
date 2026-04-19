'use client';

import * as Tabs from '@radix-ui/react-tabs';
import { useLocale } from '@/context/LocaleContext';
import { useProblemStore } from '@/store/problemStore';
import { TestCasesView } from './TestCasesView';
import { TestResultsView } from './TestResultsView';
import { SubmissionHistory } from './SubmissionHistory';
import type { Test } from '@/lib/types';

interface TestPanelProps {
  tests: Test[];
  functionName: string;
}

export function TestPanel({ tests, functionName }: TestPanelProps) {
  const { t } = useLocale();
  const { bottomTab, setBottomTab, runResult } = useProblemStore();

  return (
    <div className="flex flex-col overflow-hidden flex-[2] min-h-0" style={{ borderTop: '1px solid var(--line)', background: 'var(--bg)' }}>
      <Tabs.Root
        value={bottomTab}
        onValueChange={(v) => setBottomTab(v as 'testcases' | 'testresults' | 'submissions')}
        className="flex flex-col h-full"
      >
        <Tabs.List className="flex px-4 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
          {(['testcases', 'testresults', 'submissions'] as const).map((tab) => (
            <Tabs.Trigger
              key={tab}
              value={tab}
              className="px-3 py-2 text-sm text-text-2 transition-colors -mb-px data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:hover:text-text"
            >
              {t(tab === 'testcases' ? 'testCasesTab' : tab === 'testresults' ? 'testResultsTab' : 'submissionsTab')}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        <Tabs.Content value="testcases" className="flex-1 overflow-hidden">
          <TestCasesView tests={tests} functionName={functionName} />
        </Tabs.Content>
        <Tabs.Content value="testresults" className="flex-1 overflow-hidden">
          <TestResultsView result={runResult} tests={tests} functionName={functionName} />
        </Tabs.Content>
        <Tabs.Content value="submissions" className="flex-1 overflow-hidden">
          <SubmissionHistory />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}
