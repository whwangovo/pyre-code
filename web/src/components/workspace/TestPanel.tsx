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
    <div className="border-t border-border bg-surface flex flex-col overflow-hidden flex-[2] min-h-0">
      <Tabs.Root
        value={bottomTab}
        onValueChange={(v) => setBottomTab(v as 'testcases' | 'testresults' | 'submissions')}
        className="flex flex-col h-full"
      >
        <Tabs.List className="flex border-b border-border px-4 flex-shrink-0">
          <Tabs.Trigger
            value="testcases"
            className="px-3 py-2 text-sm data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:text-text-secondary hover:text-text-primary transition-colors -mb-px"
          >
            {t('testCasesTab')}
          </Tabs.Trigger>
          <Tabs.Trigger
            value="testresults"
            className="px-3 py-2 text-sm data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:text-text-secondary hover:text-text-primary transition-colors -mb-px"
          >
            {t('testResultsTab')}
          </Tabs.Trigger>
          <Tabs.Trigger
            value="submissions"
            className="px-3 py-2 text-sm data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:text-text-secondary hover:text-text-primary transition-colors -mb-px"
          >
            {t('submissionsTab')}
          </Tabs.Trigger>
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
