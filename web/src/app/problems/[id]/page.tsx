'use client';

import { useEffect, useState } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import * as Tabs from '@radix-ui/react-tabs';
import { Menu, ChevronLeft, ChevronRight, Sun, Moon, SwatchBook } from 'lucide-react';
import { SplitPane, VerticalSplitPane } from '@/components/ui/SplitPane';
import { ProblemDrawer } from '@/components/layout/ProblemDrawer';
import { DescriptionTab } from '@/components/workspace/DescriptionTab';
import { SolutionTab } from '@/components/workspace/SolutionTab';
import { AIHelpTab } from '@/components/workspace/AIHelpTab';
import { CodeEditor } from '@/components/workspace/CodeEditor';
import { TestPanel } from '@/components/workspace/TestPanel';
import { ActionBar } from '@/components/workspace/ActionBar';
import { useProblemStore } from '@/store/problemStore';
import { useLocale } from '@/context/LocaleContext';
import { useTheme } from '@/context/ThemeContext';
import { useDesign } from '@/context/DesignContext';
import { WorkspacePageClassic } from '@/components/workspace/WorkspacePage.classic';
import type { Problem, ProgressMap, SubmissionResult, LearningPath, LearningPathProblemSummary, SubmissionHistory } from '@/lib/types';

function FlameGlyph() {
  return (
    <span
      className="w-[22px] h-[22px] inline-flex items-center justify-center rounded-[6px] text-accent"
      style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
        <path d="M6 1.25c.6 1.8.15 2.7-.75 3.6C4 6.2 3.25 7.2 3.25 8.5a2.75 2.75 0 1 0 5.5 0c0-1-.3-1.8-1-2.6.3 1.1-.15 1.8-.8 1.8-.5 0-.85-.4-.85-1C6.1 5.6 6.7 3.9 6 1.25Z" fill="currentColor" />
      </svg>
    </span>
  );
}

export default function WorkspacePage() {
  const { design } = useDesign();
  if (design === 'classic') return <WorkspacePageClassic />;
  return <WorkspacePageNew />;
}

function WorkspacePageNew() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathId = searchParams.get('path');
  const { locale, setLocale, t, tProblem } = useLocale();
  const { theme, toggleTheme } = useTheme();
  const { toggleDesign } = useDesign();
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
      const testIndices = problem.tests.slice(0, 2).map((_, i) => i);
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: id, code: currentCode, testIndices }),
      });
      const data = await res.json();
      setRunResult(data);
      setBottomTab('testresults');
    } catch {
      setRunResult({ passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: t('networkError') });
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
        body: JSON.stringify({ taskId: id, code: currentCode }),
      });
      const data: SubmissionResult = await res.json();
      setSubmissionResult(data);
      setRunResult(data);
      setBottomTab('testresults');
      fetch('/api/progress').then((r) => r.json()).then((d) => setProgress(d.progress || {}));
      fetch(`/api/submissions/${id}`).then((r) => r.json()).then((d: SubmissionHistory[]) => setSubmissionHistory(d)).catch(() => {});
    } catch {
      const err = { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: t('networkError') };
      setSubmissionResult(err);
      setRunResult(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!problem) {
    return (
      <div className="h-screen flex items-center justify-center bg-bg">
        <p className="text-sm text-text-3">{t('loading')}</p>
      </div>
    );
  }

  // Path navigation
  const pathProblems = pathData?.problems ?? [];
  const currentPathIdx = pathProblems.findIndex((p) => p.id === id);
  const prevProblem = currentPathIdx > 0 ? pathProblems[currentPathIdx - 1] : null;
  const nextProblem = currentPathIdx >= 0 && currentPathIdx < pathProblems.length - 1 ? pathProblems[currentPathIdx + 1] : null;

  const leftPanel = (
    <div className="h-full flex flex-col bg-bg">
      <Tabs.Root defaultValue="description" className="flex flex-col h-full">
        <div
          className="flex items-center px-3.5 h-10 gap-1 flex-shrink-0"
          style={{ borderBottom: '1px solid var(--line)', background: 'var(--bg)' }}
        >
          <Tabs.List className="flex gap-0.5">
            {(['description', 'solution', 'aiHelp'] as const).map((tab) => (
              <Tabs.Trigger
                key={tab}
                value={tab}
                className="px-3 h-[26px] rounded-[6px] text-[13px] text-text-2 cursor-pointer inline-flex items-center gap-1.5 transition-[background,color] duration-150 data-[state=active]:bg-[color-mix(in_oklab,var(--text)_7%,transparent)] data-[state=active]:text-text"
              >
                {t(tab === 'description' ? 'description' : tab === 'solution' ? 'solution' : 'aiHelp')}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </div>
        <Tabs.Content value="description" className="flex-1 overflow-y-auto">
          <DescriptionTab problem={problem} />
        </Tabs.Content>
        <Tabs.Content value="solution" className="flex-1 overflow-y-auto">
          <SolutionTab problemId={id} />
        </Tabs.Content>
        <Tabs.Content value="aiHelp" className="flex-1 overflow-y-auto">
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
    <div className="h-screen flex flex-col bg-bg">
      {/* Workspace topbar */}
      <div
        className="flex items-center px-5 h-[52px] gap-4 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--line)', background: 'var(--bg)' }}
      >
        <Link href="/" className="inline-flex items-center gap-2 font-semibold text-sm tracking-[-0.01em]">
          <FlameGlyph />
          Pyre Code
        </Link>

        {/* Breadcrumbs */}
        <div className="mono text-xs text-text-3 flex items-center gap-1.5 pl-2 ml-2 h-5" style={{ borderLeft: '1px solid var(--line)' }}>
          <Link href="/problems" className="text-text-2 hover:text-text transition-colors">{t('problems')}</Link>
          <span className="opacity-60">/</span>
          <span className="text-text font-medium">{tProblem(id)}</span>
        </div>

        <div className="flex-1" />

        {/* Path progress */}
        {pathData && pathProblems.length > 0 && (
          <div
            className="flex items-center gap-2.5 px-2.5 py-1 rounded-pill mono text-xs text-text-2"
            style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          >
            <div className="flex gap-[3px]">
              {pathProblems.map((p, i) => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    background:
                      p.id === id
                        ? 'var(--accent)'
                        : p.status === 'solved'
                        ? 'var(--easy)'
                        : p.status === 'attempted'
                        ? 'var(--medium)'
                        : 'var(--line-strong)',
                    boxShadow: p.id === id ? '0 0 0 2px color-mix(in oklab, var(--accent) 25%, transparent)' : undefined,
                  }}
                />
              ))}
            </div>
            <span>{currentPathIdx + 1}/{pathProblems.length}</span>
          </div>
        )}

        {/* Nav arrows */}
        {pathData && (
          <div className="flex gap-1">
            <button
              onClick={() => prevProblem && router.push(`/problems/${prevProblem.id}?path=${pathId}`)}
              disabled={!prevProblem}
              className="w-7 h-7 inline-flex items-center justify-center rounded-[7px] text-text-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => nextProblem && router.push(`/problems/${nextProblem.id}?path=${pathId}`)}
              disabled={!nextProblem}
              className="w-7 h-7 inline-flex items-center justify-center rounded-[7px] text-text-2 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
              style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        <button
          onClick={() => setDrawerOpen(true)}
          className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          <Menu className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={toggleDesign}
          className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
          title="Switch design"
        >
          <SwatchBook className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={toggleTheme}
          className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>

        <button
          onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}
          className="h-[30px] px-3 inline-flex items-center rounded-lg text-[13px] text-text-2 cursor-pointer hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          {locale === 'en' ? 'EN' : '中文'}
        </button>
      </div>

      <div className="flex-1 overflow-hidden" style={{ background: 'var(--bg-sunken)' }}>
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
