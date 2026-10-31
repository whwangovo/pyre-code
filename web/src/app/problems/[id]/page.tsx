'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import * as Tabs from '@radix-ui/react-tabs';
import { Menu, ChevronLeft, ChevronRight, Sun, Moon } from 'lucide-react';
import { SplitPane, VerticalSplitPane } from '@/components/ui/SplitPane';
import { ProblemDrawer } from '@/components/layout/ProblemDrawer';
import { FeedbackModal } from '@/components/workspace/FeedbackModal';
import { DescriptionTab } from '@/components/workspace/DescriptionTab';
import { KnowledgeTab } from '@/components/workspace/KnowledgeTab';
import { SolutionTab } from '@/components/workspace/SolutionTab';
import { AIHelpTab } from '@/components/workspace/AIHelpTab';
import { SubmissionHistory as SubmissionHistoryTab } from '@/components/workspace/SubmissionHistory';
import { CodeEditor } from '@/components/workspace/CodeEditor';
import { TestPanel } from '@/components/workspace/TestPanel';
import { ActionBar } from '@/components/workspace/ActionBar';
import { useProblemStore } from '@/store/problemStore';
import { useLocale } from '@/context/LocaleContext';
import { useTheme } from '@/context/ThemeContext';
import type { Problem, ProgressMap, SubmissionResult, LearningPath, LearningPathProblemSummary, SubmissionHistory, CodeMode } from '@/lib/types';
import { flushCodeDraft, loadCodeMode, resolveCodeDraft, saveCodeDraft, saveCodeMode } from '@/lib/codeDraft';
import { loadAiChat } from '@/lib/aiChatHistory';
import { cn } from '@/lib/utils';

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
  return <WorkspacePageNew />;
}

function WorkspacePageNew() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathId = searchParams.get('path');
  const { locale, setLocale, t } = useLocale();
  const { theme, toggleTheme } = useTheme();
  const {
    currentCode, setCurrentCode,
    submissionResult, setSubmissionResult,
    isSubmitting, setIsSubmitting,
    drawerOpen, setDrawerOpen,
    isRunning, setIsRunning,
    setRunResult, setBottomTab, resetTestPanel, resetAiHelp, setAiHelpMessages,
    submissionHistory, setSubmissionHistory,
    codeMode, setCodeMode,
  } = useProblemStore();

  const [problem, setProblem] = useState<(Problem & { starterCode?: string }) | null>(null);
  const [allProblems, setAllProblems] = useState<Problem[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [pathData, setPathData] = useState<(Omit<LearningPath, 'problems'> & { problems: LearningPathProblemSummary[] }) | null>(null);
  const [feedbackResult, setFeedbackResult] = useState<SubmissionResult | null>(null);
  // Set when the problem can't be shown: an unknown id (old link, typo) or a backend error
  const [loadError, setLoadError] = useState<'notFound' | 'failed' | null>(null);
  const codeReadyRef = useRef(false);
  // The code last loaded or saved, so only real edits get saved
  const savedCodeRef = useRef<string | null>(null);
  // Bumped on every code load and on leaving the problem; a load that finishes after it
  // changed is stale and must not put its code into the editor (which would save it)
  const loadSeqRef = useRef(0);

  useEffect(() => {
    const seq = ++loadSeqRef.current;
    const isStale = () => seq !== loadSeqRef.current;
    codeReadyRef.current = false;
    setLoadError(null);
    fetch(`/api/problems/${id}`)
      .then(async (r) => {
        if (!r.ok) {
          if (!isStale()) setLoadError(r.status === 404 ? 'notFound' : 'failed');
          return;
        }
        const data: Problem & { starterCode?: string } = await r.json();
        if (isStale()) return;
        // Hot 100 remembers the learner's answer mode; AI problems are always core
        const mode: CodeMode = data.starters ? loadCodeMode() : 'core';
        const cachedCode = await resolveCodeDraft(id, mode);
        if (isStale()) return;
        setProblem(data);
        setCodeMode(mode);
        const code = cachedCode ?? data.starters?.[mode] ?? data.starterCode ?? '';
        savedCodeRef.current = code;
        setCurrentCode(code);
        codeReadyRef.current = true;
        setSubmissionResult(null);
        resetTestPanel();
      })
      .catch(() => { if (!isStale()) setLoadError('failed'); });
    fetch('/api/problems')
      .then((r) => r.json())
      .then((d) => setAllProblems(d.problems))
      .catch(() => {});
    fetch('/api/progress')
      .then((r) => r.json())
      .then((d) => setProgress(d.progress || {}))
      .catch(() => {});
    // Cleared up front (not in resetTestPanel after the problem loads), so a history
    // response that beats the problem response isn't wiped
    setSubmissionHistory([]);
    fetch(`/api/submissions/${id}`)
      .then((r) => r.json())
      .then((d: SubmissionHistory[]) => { if (!isStale()) setSubmissionHistory(d); })
      .catch(() => {});
    // Same for the AI chat: reset up front, then show this problem's saved conversation
    resetAiHelp();
    loadAiChat(id).then((messages) => {
      // Skip if the learner already started chatting before the history arrived
      const { aiHelpMessages, aiHelpLoading } = useProblemStore.getState();
      if (!isStale() && aiHelpMessages.length === 0 && !aiHelpLoading) setAiHelpMessages(messages);
    });
    if (pathId) {
      fetch(`/api/paths/${pathId}`)
        .then((r) => r.json())
        .then((d) => setPathData(d));
    } else {
      setPathData(null);
    }
    return () => { loadSeqRef.current++; };
  }, [id, pathId, setCurrentCode, setCodeMode, setSubmissionResult, resetTestPanel, resetAiHelp, setAiHelpMessages, setSubmissionHistory]);

  // Keyboard shortcuts: [ / ] for prev/next problem, Esc to close drawer.
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLElement &&
        (e.target.tagName === 'INPUT' ||
          e.target.tagName === 'TEXTAREA' ||
          e.target.isContentEditable ||
          e.target.closest('.monaco-editor'))
      )
        return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === 'Escape' && drawerOpen) {
        setDrawerOpen(false);
        e.preventDefault();
        return;
      }

      const pathProblems = pathData?.problems ?? [];
      const currentIdx = pathProblems.findIndex((p) => p.id === id);
      const prev = currentIdx > 0 ? pathProblems[currentIdx - 1] : null;
      const next =
        currentIdx >= 0 && currentIdx < pathProblems.length - 1
          ? pathProblems[currentIdx + 1]
          : null;

      if (e.key === '[' && prev) {
        router.push(`/problems/${prev.id}?path=${pathId}`);
        e.preventDefault();
      } else if (e.key === ']' && next) {
        router.push(`/problems/${next.id}?path=${pathId}`);
        e.preventDefault();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [pathData, id, pathId, router, drawerOpen, setDrawerOpen]);

  // Also follows mode changes, so restoring an ACM submission saves into the ACM draft.
  // Freshly loaded code isn't saved back: that would stamp it as the newest edit and
  // could overwrite a newer draft written on another device.
  useEffect(() => {
    if (!codeReadyRef.current || currentCode === savedCodeRef.current) return;
    savedCodeRef.current = currentCode;
    saveCodeDraft(id, currentCode, codeMode);
  }, [id, currentCode, codeMode]);

  // Push the last edit to the server when leaving the problem or hiding/closing the tab
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState === 'hidden') flushCodeDraft();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    window.addEventListener('pagehide', flushCodeDraft);
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange);
      window.removeEventListener('pagehide', flushCodeDraft);
      flushCodeDraft();
    };
  }, []);

  const isHot100 = problem?.track === 'hot100';
  const mode = isHot100 ? codeMode : undefined;

  const switchCodeMode = async (next: CodeMode) => {
    if (!problem?.starters || next === codeMode) return;
    const starters = problem.starters;
    const seq = ++loadSeqRef.current;
    codeReadyRef.current = false;
    saveCodeMode(next);
    setCodeMode(next);
    setRunResult(null);
    const code = (await resolveCodeDraft(id, next)) ?? starters[next];
    if (seq !== loadSeqRef.current) return;
    savedCodeRef.current = code;
    setCurrentCode(code);
    codeReadyRef.current = true;
  };

  const handleRun = async () => {
    if (!problem || isRunning) return;
    setIsRunning(true);
    setRunResult(null);
    try {
      const testIndices = problem.tests.slice(0, 2).map((_, i) => i);
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: id, code: currentCode, testIndices, mode }),
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
        body: JSON.stringify({ taskId: id, code: currentCode, mode }),
      });
      const data: SubmissionResult = await res.json();
      setSubmissionResult(data);
      setRunResult(data);
      setBottomTab('testresults');
      setFeedbackResult(data);
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

  // Keyboard shortcuts: Cmd/Ctrl+Enter runs, Cmd/Ctrl+Shift+Enter submits.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!((e.metaKey || e.ctrlKey) && e.key === 'Enter')) return;
      if (isRunning || isSubmitting) return;
      e.preventDefault();
      if (e.shiftKey) {
        handleSubmit();
      } else {
        handleRun();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handleRun, handleSubmit, isRunning, isSubmitting]);

  if (loadError) {
    return (
      <div className="h-screen flex flex-col items-center justify-center gap-3 bg-bg">
        <p className="text-sm text-text-2">{t(loadError === 'notFound' ? 'problemNotFound' : 'problemLoadFailed')}</p>
        <Link href="/problems" className="text-sm text-accent hover:underline">{t('backToProblems')}</Link>
      </div>
    );
  }

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
            {/* Concept notes are written for the AI problems only */}
            {(['description', ...(isHot100 ? [] : ['knowledge'] as const), 'solution', 'aiHelp', 'submissionsTab'] as const).map((tab) => (
              <Tabs.Trigger
                key={tab}
                value={tab}
                className="group px-3 h-[26px] rounded-[6px] text-[13px] text-text-2 cursor-pointer inline-flex items-center gap-1.5 transition-[background,color] duration-150 data-[state=active]:bg-[color-mix(in_oklab,var(--text)_7%,transparent)] data-[state=active]:text-text"
              >
                <span className="w-[5px] h-[5px] rounded-full bg-accent hidden group-data-[state=active]:inline-block" />
                {t(tab)}
                {tab === 'submissionsTab' && submissionHistory.length > 0 && (
                  <span className="text-[11px] text-text-3 tabular-nums">{submissionHistory.length}</span>
                )}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </div>
        <Tabs.Content value="description" className="flex-1 overflow-y-auto">
          <DescriptionTab problem={problem} />
        </Tabs.Content>
        <Tabs.Content value="knowledge" className="flex-1 overflow-y-auto">
          <KnowledgeTab problemId={id} />
        </Tabs.Content>
        <Tabs.Content value="solution" className="flex-1 overflow-y-auto">
          <SolutionTab problemId={id} />
        </Tabs.Content>
        <Tabs.Content value="aiHelp" className="flex-1 min-h-0">
          {/* Keyed so the unsent message reloads for the new problem */}
          <AIHelpTab key={problem.id} problem={problem} />
        </Tabs.Content>
        <Tabs.Content value="submissionsTab" className="flex-1 min-h-0">
          <SubmissionHistoryTab showMode={isHot100} />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );

  const rightPanel = (
    <div className="h-full flex flex-col">
      <VerticalSplitPane
        top={
          <div className="flex flex-col h-full">
            <div
              className="flex items-center gap-2.5 px-3.5 flex-shrink-0"
              style={{
                height: '36px',
                borderBottom: '1px solid var(--line)',
                background: 'color-mix(in oklab, var(--text) 2%, var(--bg))',
              }}
            >
              <svg className="w-3.5 h-3.5 text-text-3" viewBox="0 0 16 16" fill="none">
                <path d="M3 1.5h7l3 3V14.5H3V1.5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
                <path d="M10 1.5v3h3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
              </svg>
              <span className="mono text-[12.5px] text-text-2">{id}.py</span>
              {isHot100 && (
                <div className="ml-auto flex gap-0.5 p-0.5 rounded-[7px]" style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}>
                  {(['core', 'acm'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => switchCodeMode(m)}
                      title={m === 'core' ? t('modeCoreTip') : t('modeAcmTip')}
                      className={cn(
                        'px-2 h-[20px] rounded-[5px] text-[11.5px] cursor-pointer transition-[background,color] duration-150',
                        codeMode === m ? 'bg-bg-elev text-text shadow-sm' : 'text-text-3 hover:text-text',
                      )}
                    >
                      {m === 'core' ? t('modeCore') : t('modeAcm')}
                    </button>
                  ))}
                </div>
              )}
              <span className={cn(!isHot100 && 'ml-auto', 'mono text-[11.5px] text-text-3 px-1.5 py-0.5 rounded')} style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}>Python</span>
            </div>
            <div className="flex-1 overflow-hidden">
              <CodeEditor value={currentCode} onChange={setCurrentCode} onRunShortcut={handleRun} onSubmitShortcut={handleSubmit} />
            </div>
          </div>
        }
        bottom={
          <div className="flex flex-col h-full">
            <TestPanel tests={problem.tests} functionName={problem.functionName} cases={problem.cases?.[codeMode]} codeMode={codeMode} />
            <ActionBar onSubmit={handleSubmit} onRun={handleRun} isSubmitting={isSubmitting} isRunning={isRunning} attemptCount={submissionHistory.length} />
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
          <Link href={isHot100 ? '/hot100' : '/problems'} className="text-text-2 hover:text-text transition-colors">
            {isHot100 ? t('hot100') : t('problems')}
          </Link>
          <span className="opacity-60">/</span>
          <span className="text-text font-medium">{problem ? (locale === 'zh' ? problem.titleZh : problem.title) : id}</span>
        </div>

        <button
          onClick={() => setDrawerOpen(true)}
          className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-text-2 cursor-pointer hover:text-text hover:border-line-strong transition-[color,border-color] duration-150"
          style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}
        >
          <Menu className="w-3.5 h-3.5" />
        </button>

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
        problems={allProblems.filter((p) => p.track === problem.track)}
        progress={progress}
        currentId={id}
      />
      {feedbackResult && (
        <FeedbackModal
          result={feedbackResult}
          onClose={() => setFeedbackResult(null)}
          attemptNumber={submissionHistory.length}
          pathProgress={pathData ? {
            solved: pathData.problems.filter(p => p.status === 'solved').length,
            total: pathData.problems.length,
            name: locale === 'zh' ? pathData.titleZh : pathData.titleEn,
          } : null}
        />
      )}
    </div>
  );
}
