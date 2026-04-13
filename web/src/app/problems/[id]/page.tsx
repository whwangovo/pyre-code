'use client';

import { useEffect, useState } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import * as Tabs from '@radix-ui/react-tabs';
import { Menu, ChevronLeft, ChevronRight } from 'lucide-react';
import { SplitPane, VerticalSplitPane } from '@/components/ui/SplitPane';
import { TopNav } from '@/components/layout/TopNav';
import { ProblemDrawer } from '@/components/layout/ProblemDrawer';
import { DescriptionTab } from '@/components/workspace/DescriptionTab';
import { SolutionTab } from '@/components/workspace/SolutionTab';
import { AIHelpTab } from '@/components/workspace/AIHelpTab';
import { CodeEditor } from '@/components/workspace/CodeEditor';
import { TestPanel } from '@/components/workspace/TestPanel';
import { ActionBar } from '@/components/workspace/ActionBar';
import { useProblemStore } from '@/store/problemStore';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, ProgressMap, SubmissionResult, LearningPath, LearningPathProblemSummary, SubmissionHistory } from '@/lib/types';

export default function WorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathId = searchParams.get('path');
  const { t, tProblem } = useLocale();
  const {
    currentCode, setCurrentCode,
    submissionResult, setSubmissionResult,
    isSubmitting, setIsSubmitting,
    drawerOpen, setDrawerOpen,
    isRunning, setIsRunning,
    setRunResult, setBottomTab, resetTestPanel, resetAiHelp,
    submissionHistory, setSubmissionHistory,
  } = useProblemStore();

  const [problem, setProblem] = useState<(Problem & { starterCode?: string }) | null>(null);
  const [allProblems, setAllProblems] = useState<Problem[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [pathData, setPathData] = useState<(Omit<LearningPath, 'problems'> & { problems: LearningPathProblemSummary[] }) | null>(null);

  useEffect(() => {
    fetch(`/api/problems/${id}`)
      .then((r) => r.json())
      .then((data) => {
        setProblem(data);
        setCurrentCode(data.starterCode || '');
        setSubmissionResult(null);
        resetTestPanel();
        resetAiHelp();
      });
    fetch('/api/problems')
      .then((r) => r.json())
      .then((d) => setAllProblems(d.problems));
    fetch('/api/progress')
      .then((r) => r.json())
      .then((d) => setProgress(d.progress || {}));
    fetch(`/api/submissions/${id}`)
      .then((r) => r.json())
      .then((d: SubmissionHistory[]) => setSubmissionHistory(d))
      .catch(() => {});
    if (pathId) {
      fetch(`/api/paths/${pathId}`)
        .then((r) => r.json())
        .then((d) => setPathData(d));
    } else {
      setPathData(null);
    }
  }, [id, pathId, setCurrentCode, setSubmissionResult, resetTestPanel, resetAiHelp, setSubmissionHistory]);

  const handleRun = async () => {
    if (!problem || isRunning) return;
    setIsRunning(true);
    setRunResult(null);
    try {
      // Run only first 2 sample tests
      const testIndices = problem.tests.slice(0, 2).map((_, i) => i);
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: problem.id, code: currentCode, testIndices }),
      });
      const result: SubmissionResult = await res.json();
      setRunResult(result);
      setBottomTab('testresults');
    } catch {
      setRunResult({ passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: 'Network error' });
      setBottomTab('testresults');
    } finally {
      setIsRunning(false);
    }
  };

  const handleSubmit = async () => {
    if (!problem || isSubmitting) return;
    setIsSubmitting(true);
    setSubmissionResult(null);
    try {
      const res = await fetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: problem.id, code: currentCode }),
      });
      const result: SubmissionResult = await res.json();
      setSubmissionResult(result);
      setRunResult(result);
      setBottomTab('testresults');
      // Prepend to submission history
      const newEntry: SubmissionHistory = {
        id: Date.now(),
        passed: result.allPassed,
        execTimeMs: result.totalTimeMs,
        submittedAt: new Date().toISOString(),
        code: currentCode,
      };
      setSubmissionHistory([newEntry, ...submissionHistory]);
      // Refresh progress
      const progRes = await fetch('/api/progress');
      const progData = await progRes.json();
      setProgress(progData.progress || {});
    } catch (e) {
      const errResult: SubmissionResult = { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: 'Network error' };
      setSubmissionResult(errResult);
      setRunResult(errResult);
      setBottomTab('testresults');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!problem) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <p className="text-sm text-text-tertiary">{t('loading')}</p>
      </div>
    );
  }


  // Path navigation
  const pathProblemIds = pathData?.problems?.map((p) => p.id) ?? [];
  const pathIdx = pathProblemIds.indexOf(id);
  const prevPathId = pathIdx > 0 ? pathProblemIds[pathIdx - 1] : null;
  const nextPathId = pathIdx >= 0 && pathIdx < pathProblemIds.length - 1 ? pathProblemIds[pathIdx + 1] : null;

  const leftPanel = (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-border">
        <button
          onClick={() => setDrawerOpen(true)}
          className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <Menu className="w-4 h-4 text-text-secondary" />
        </button>
        <div className="flex-1 min-w-0">
          {pathData && (
            <div className="flex items-center gap-1 mb-0.5">
              <Link href={`/paths/${pathData.id}`} className="text-xs text-accent hover:underline truncate">
                {pathData.titleEn}
              </Link>
              <span className="text-xs text-text-tertiary">·</span>
              <span className="text-xs text-text-tertiary">{pathIdx + 1}/{pathProblemIds.length}</span>
            </div>
          )}
          <span className="text-sm font-medium text-text-primary truncate block">{tProblem(problem.id)}</span>
        </div>
        {pathData && (
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={() => prevPathId && router.push(`/problems/${prevPathId}?path=${pathId}`)}
              disabled={!prevPathId}
              className="p-1 rounded hover:bg-gray-100 disabled:opacity-30 transition-colors"
              title={t('prevProblem')}
            >
              <ChevronLeft className="w-4 h-4 text-text-secondary" />
            </button>
            <button
              onClick={() => nextPathId && router.push(`/problems/${nextPathId}?path=${pathId}`)}
              disabled={!nextPathId}
              className="p-1 rounded hover:bg-gray-100 disabled:opacity-30 transition-colors"
              title={t('nextProblem')}
            >
              <ChevronRight className="w-4 h-4 text-text-secondary" />
            </button>
          </div>
        )}
      </div>
      <Tabs.Root defaultValue="description" className="flex-1 flex flex-col overflow-hidden">
        <Tabs.List className="flex border-b border-border px-4">
          <Tabs.Trigger
            value="description"
            className="px-3 py-2 text-sm data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:text-text-secondary hover:text-text-primary transition-colors -mb-px"
          >
            {t('description')}
          </Tabs.Trigger>
          <Tabs.Trigger
            value="solution"
            className="px-3 py-2 text-sm data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:text-text-secondary hover:text-text-primary transition-colors -mb-px"
          >
            {t('solution')}
          </Tabs.Trigger>
          <Tabs.Trigger
            value="ai-help"
            className="px-3 py-2 text-sm data-[state=active]:text-accent data-[state=active]:border-b-2 data-[state=active]:border-accent data-[state=inactive]:text-text-secondary hover:text-text-primary transition-colors -mb-px"
          >
            {t('aiHelp')}
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="description" className="flex-1 overflow-auto">
          <DescriptionTab problem={problem} />
        </Tabs.Content>
        <Tabs.Content value="solution" className="flex-1 overflow-auto">
          <SolutionTab problemId={problem.id} />
        </Tabs.Content>
        <Tabs.Content value="ai-help" className="flex-1 overflow-auto">
          <AIHelpTab problem={problem} />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );

  const rightPanel = (
    <div className="h-full flex flex-col">
      <VerticalSplitPane
        top={<CodeEditor value={currentCode} onChange={setCurrentCode} />}
        bottom={
          <div className="flex flex-col h-full">
            <TestPanel tests={problem.tests} functionName={problem.functionName} />
            <ActionBar onSubmit={handleSubmit} onRun={handleRun} isSubmitting={isSubmitting} isRunning={isRunning} />
          </div>
        }
        defaultRatio={0.65}
        minTop={200}
        minBottom={150}
      />
    </div>
  );

  return (
    <div className="h-screen flex flex-col bg-surface">
      <TopNav />
      <div className="flex-1 overflow-hidden">
        <SplitPane left={leftPanel} right={rightPanel} />
      </div>
      <ProblemDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        problems={allProblems}
        progress={progress}
        currentId={id}
      />
    </div>
  );
}
