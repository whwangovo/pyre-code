'use client';

import { useEffect, useRef } from 'react';
import { X, CheckCircle, XCircle, AlertCircle, AlertTriangle, Sparkles } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import type { TranslationKey } from '@/lib/i18n';
import type { SubmissionResult } from '@/lib/types';

type FeedbackKind = 'accepted' | 'wrong' | 'runtime' | 'compile';

interface FeedbackModalProps {
  result: SubmissionResult;
  onClose: () => void;
  attemptNumber?: number;
  pathProgress?: { solved: number; total: number; name: string } | null;
  /** Opens the AI Help tab; the "View Hint" offer is shown only when provided */
  onViewHint?: () => void;
}

// Messages produced before any test runs: Python's SyntaxError / IndentationError and the
// grading service's static check ("Syntax error: ...", "... allowed at the top level")
const COMPILE_MARKERS = ['syntaxerror', 'syntax error', 'indentationerror', 'indentation', 'allowed at the top level'];
// Exceptions arrive as "SomeError: ..." or "SomeException: ..."; the judge itself reports
// "Runtime Error" and "Time Limit Exceeded". A failed assertion carries only its own message.
const RUNTIME_MARKER = /^(\w*(Error|Exception)\b|Runtime Error|Time Limit Exceeded)/;

function isCompileError(message: string): boolean {
  const text = message.toLowerCase();
  return COMPILE_MARKERS.some((marker) => text.includes(marker));
}

function classifyResult(result: SubmissionResult): FeedbackKind {
  if (result.allPassed) return 'accepted';
  if (result.error) return isCompileError(result.error) ? 'compile' : 'runtime';
  const firstErr = result.results.find((r) => r.error)?.error ?? '';
  if (isCompileError(firstErr)) return 'compile';
  if (RUNTIME_MARKER.test(firstErr)) return 'runtime';
  return 'wrong';
}

const kindConfig: Record<FeedbackKind, {
  icon: typeof CheckCircle;
  tagKey: TranslationKey;
  titleKey: TranslationKey;
  headClass: string;
  iconClass: string;
  titleColor: string;
}> = {
  accepted: {
    icon: CheckCircle,
    tagKey: 'feedbackAccepted',
    titleKey: 'feedbackAcceptedTitle',
    headClass: 'bg-gradient-to-b from-[color-mix(in_oklab,var(--easy)_10%,var(--bg-elev))] to-[var(--bg-elev)]',
    iconClass: 'bg-easy text-white',
    titleColor: 'text-easy',
  },
  wrong: {
    icon: XCircle,
    tagKey: 'feedbackWrong',
    titleKey: 'feedbackWrongTitle',
    headClass: 'bg-gradient-to-b from-[color-mix(in_oklab,var(--hard)_8%,var(--bg-elev))] to-[var(--bg-elev)]',
    iconClass: 'border border-[color-mix(in_oklab,var(--hard)_35%,var(--line))] bg-[color-mix(in_oklab,var(--hard)_16%,var(--bg-elev))] text-hard',
    titleColor: 'text-hard',
  },
  runtime: {
    icon: AlertCircle,
    tagKey: 'feedbackRuntime',
    titleKey: 'feedbackRuntimeTitle',
    headClass: 'bg-gradient-to-b from-[color-mix(in_oklab,var(--hard)_10%,var(--bg-elev))] to-[var(--bg-elev)]',
    iconClass: 'bg-[color-mix(in_oklab,var(--hard)_90%,var(--bg-elev))] text-white',
    titleColor: 'text-hard',
  },
  compile: {
    icon: AlertTriangle,
    tagKey: 'feedbackCompile',
    titleKey: 'feedbackCompileTitle',
    headClass: 'bg-gradient-to-b from-[color-mix(in_oklab,var(--medium)_10%,var(--bg-elev))] to-[var(--bg-elev)]',
    iconClass: 'border border-[color-mix(in_oklab,var(--medium)_35%,var(--line))] bg-[color-mix(in_oklab,var(--medium)_16%,var(--bg-elev))] text-medium',
    titleColor: 'text-medium',
  },
};

function AcceptedBody({ result, pathProgress }: { result: SubmissionResult; pathProgress?: { solved: number; total: number; name: string } | null }) {
  const { t } = useLocale();
  const totalMs = result.totalTimeMs;
  const slowestMs = result.results.reduce((max, r) => Math.max(max, r.execTimeMs), 0);

  return (
    <>
      {/* Only what the judge measured: total and slowest test time, tests passed */}
      <div className="grid grid-cols-3 gap-2.5 mb-3.5">
        {[
          { k: t('feedbackStatRuntime'), v: `${totalMs.toFixed(0)}`, u: 'ms' },
          { k: t('feedbackStatSlowest'), v: `${slowestMs.toFixed(0)}`, u: 'ms' },
          { k: t('feedbackStatPassed'), v: `${result.passed}/${result.total}`, u: '' },
        ].map(({ k, v, u }) => (
          <div key={k} className="p-3 rounded-[9px] border border-[var(--line)]" style={{ background: 'var(--bg-sunken)' }}>
            <div className="font-mono text-[10px] tracking-[.12em] uppercase text-text-3">{k}</div>
            <div className="text-xl mt-0.5 tabular-nums text-text tracking-tight">
              {v}{u && <span className="text-text-3 text-xs font-normal ml-0.5">{u}</span>}
            </div>
          </div>
        ))}
      </div>

      {pathProgress && pathProgress.total > 0 && (
        <div
          className="flex gap-3 items-center p-3 mt-3.5 rounded-[10px] border border-[var(--accent-line)]"
          style={{ background: 'var(--accent-wash)' }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-accent"
            style={{ background: 'color-mix(in oklab, var(--accent) 18%, var(--bg-elev))' }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[13.5px] font-medium text-text truncate">{pathProgress.name}</div>
            <div className="font-mono text-[11px] text-text-3">{pathProgress.solved}/{pathProgress.total}</div>
          </div>
          <div className="w-[68px] h-1 rounded-full relative overflow-hidden" style={{ background: 'var(--line)' }}>
            <div
              className="absolute inset-0 rounded-full bg-accent"
              style={{ width: `${(pathProgress.solved / pathProgress.total) * 100}%` }}
            />
          </div>
        </div>
      )}
    </>
  );
}

function parseDiffFromError(error: string): { expected?: string; got?: string } | null {
  const expMatch = error.match(/expected[:\s]+(.+)/i);
  const gotMatch = error.match(/got[:\s]+(.+)/i);
  if (expMatch && gotMatch) return { expected: expMatch[1].trim(), got: gotMatch[1].trim() };
  const assertMatch = error.match(/AssertionError:\s*(.+?)\s*!=\s*(.+)/);
  if (assertMatch) return { expected: assertMatch[2].trim(), got: assertMatch[1].trim() };
  const lines = error.split('\n').filter(l => l.trim());
  if (lines.length >= 2) return { expected: lines[lines.length - 2].trim(), got: lines[lines.length - 1].trim() };
  return null;
}

function WrongBody({ result, onViewHint }: { result: SubmissionResult; onViewHint?: () => void }) {
  const { t } = useLocale();
  const firstFail = result.results.find(r => !r.passed);
  const diff = firstFail?.error ? parseDiffFromError(firstFail.error) : null;

  return (
    <>
      {/* Test mini pills */}
      <div className="flex gap-1 mb-2.5 flex-wrap">
        {result.results.map((r, i) => (
          <span
            key={i}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-[11px] border ${
              r.passed
                ? 'text-easy border-[color-mix(in_oklab,var(--easy)_28%,var(--line))] bg-[color-mix(in_oklab,var(--easy)_6%,var(--bg-sunken))]'
                : 'text-hard border-[color-mix(in_oklab,var(--hard)_28%,var(--line))] bg-[color-mix(in_oklab,var(--hard)_6%,var(--bg-sunken))]'
            }`}
          >
            {r.passed ? '✓' : '✕'} {r.name}
          </span>
        ))}
      </div>

      {/* Diff comparison */}
      {diff ? (
        <div className="rounded-[10px] border border-[var(--line)] overflow-hidden mb-2.5 font-mono text-xs">
          <div
            className="grid py-1.5 px-3"
            style={{ gridTemplateColumns: '64px 1fr', background: 'color-mix(in oklab, var(--easy) 6%, var(--bg-sunken))' }}
          >
            <span className="text-[10.5px] tracking-[.12em] text-text-3 pt-px">EXPECTED</span>
            <span style={{ color: 'color-mix(in oklab, var(--easy) 55%, var(--text))' }}>{diff.expected}</span>
          </div>
          <div
            className="grid py-1.5 px-3"
            style={{ gridTemplateColumns: '64px 1fr', borderTop: '1px solid var(--line)', background: 'color-mix(in oklab, var(--hard) 6%, var(--bg-sunken))' }}
          >
            <span className="text-[10.5px] tracking-[.12em] text-text-3 pt-px">GOT</span>
            <span style={{ color: 'color-mix(in oklab, var(--hard) 55%, var(--text))' }}>{diff.got}</span>
          </div>
        </div>
      ) : firstFail?.error ? (
        <pre
          className="font-mono text-xs leading-relaxed p-3 rounded-[9px] border border-[var(--line)] overflow-auto whitespace-pre-wrap break-words mb-2.5"
          style={{ background: 'var(--bg-sunken)' }}
        >
          {firstFail.error}
        </pre>
      ) : null}

      {/* Hint offer */}
      {onViewHint && (
        <div
          className="flex items-center gap-3 p-3 rounded-[10px] border border-dashed border-[var(--line)] mt-2.5"
          style={{ background: 'var(--bg-sunken)' }}
        >
          <Sparkles className="w-[18px] h-[18px] text-accent flex-shrink-0" />
          <div className="flex-1 text-[13px] text-text-2">
            <span className="font-medium text-text">{t('feedbackNeedHint')}</span>{' '}
            {t('feedbackHintBody')}
          </div>
          <button
            type="button"
            onClick={onViewHint}
            className="h-7 px-3 text-xs rounded-[7px] cursor-pointer text-accent flex-shrink-0"
            style={{ border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }}
          >
            {t('feedbackViewHint')}
          </button>
        </div>
      )}
    </>
  );
}

function parseTraceback(error: string): { files: { path: string; code: string }[]; message: string } {
  const lines = error.split('\n');
  const files: { path: string; code: string }[] = [];
  let message = '';
  for (let i = 0; i < lines.length; i++) {
    const fileLine = lines[i].match(/^\s*File\s+"?(.+?)"?,\s*line\s*(\d+)/i);
    if (fileLine) {
      const code = (i + 1 < lines.length && !lines[i + 1].match(/^\s*File/)) ? lines[i + 1].trim() : '';
      files.push({ path: `${fileLine[1]}:${fileLine[2]}`, code });
      if (code) i++;
    } else if (lines[i].match(/Error[:\s]/i) && i === lines.length - 1) {
      message = lines[i].trim();
    } else if (i === lines.length - 1 && !message) {
      message = lines[i].trim();
    }
  }
  if (!message && lines.length > 0) message = lines[lines.length - 1].trim();
  return { files, message };
}

function RuntimeBody({ result }: { result: SubmissionResult }) {
  const { t } = useLocale();
  const errorText = result.error || result.results.find(r => r.error)?.error || '';
  const tb = parseTraceback(errorText);

  return (
    <>
      {/* Structured traceback */}
      <div
        className="font-mono text-xs leading-[1.7] p-3.5 rounded-[9px] border border-[var(--line)] overflow-auto mb-2.5"
        style={{ background: 'var(--bg-sunken)' }}
      >
        {tb.files.length > 0 ? (
          <>
            {tb.files.map((f, i) => (
              <div key={i}>
                <div className="text-text-3">{f.path}</div>
                {f.code && <div className="text-text pl-4">{f.code}</div>}
              </div>
            ))}
            {tb.message && (
              <div
                className="text-hard font-medium mt-2.5 pt-2.5"
                style={{ borderTop: '1px solid var(--line)' }}
              >
                {tb.message}
              </div>
            )}
          </>
        ) : (
          <div className="text-hard font-medium whitespace-pre-wrap">{errorText}</div>
        )}
      </div>

      {/* Causes list */}
      <h4 className="font-mono text-[10.5px] tracking-[.12em] uppercase text-text-3 font-medium mt-4 mb-2">
        {t('feedbackLikelyCauses')}
      </h4>
      <div className="rounded-[9px] border border-[var(--line)] overflow-hidden" style={{ background: 'var(--bg-elev)' }}>
        <ul className="list-none p-0 m-0">
          {getLikelyCauses(errorText).map((causeKey, i) => (
            <li
              key={i}
              className="flex items-start gap-2.5 py-2 px-3 text-[13px] text-text-2"
              style={{ borderTop: i > 0 ? '1px solid var(--line)' : undefined }}
            >
              <span
                className="w-[18px] h-[18px] rounded-[5px] flex items-center justify-center flex-shrink-0 font-mono text-[10px] text-text-3"
                style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
              >
                {i + 1}
              </span>
              <span>{t(causeKey)}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

function parseCompileError(error: string): { lines: { num: number; code: string; hit: boolean }[]; pointer?: string; message: string } {
  const parts = error.split('\n');
  const lines: { num: number; code: string; hit: boolean }[] = [];
  let pointer = '';
  let message = '';

  for (let i = 0; i < parts.length; i++) {
    const lineMatch = parts[i].match(/^\s*line\s+(\d+)/i);
    if (lineMatch && !message) {
      message = parts[i].trim();
      continue;
    }
    if (parts[i].match(/^\s*\^+/)) {
      pointer = parts[i];
      continue;
    }
    if (parts[i].match(/Error[:\s]/i)) {
      message = parts[i].trim();
      continue;
    }
  }

  // Try to extract code context from error
  const codeLines = parts.filter(l => !l.match(/^\s*\^/) && !l.match(/Error[:\s]/i) && !l.match(/^\s*File/) && !l.match(/^\s*line\s+\d+/i) && l.trim());
  const lineNumMatch = error.match(/line\s+(\d+)/i);
  const startLine = lineNumMatch ? Math.max(1, parseInt(lineNumMatch[1]) - 1) : 1;

  if (codeLines.length > 0) {
    codeLines.forEach((code, i) => {
      const num = startLine + i;
      const isHit = lineNumMatch ? num === parseInt(lineNumMatch[1]) : i === codeLines.length - 1;
      lines.push({ num, code: code.trim(), hit: isHit });
    });
  }

  if (!message) message = parts[parts.length - 1]?.trim() || error.trim();
  return { lines, pointer, message };
}

function CompileBody({ result }: { result: SubmissionResult }) {
  const { t } = useLocale();
  const errorText = result.error || result.results.find(r => r.error)?.error || '';
  const parsed = parseCompileError(errorText);

  return (
    <>
      {/* Line error block */}
      <div
        className="font-mono text-[12.5px] leading-[1.65] p-3.5 rounded-[9px] border border-[var(--line)] overflow-auto mb-2.5"
        style={{ background: 'var(--bg-sunken)' }}
      >
        {parsed.lines.length > 0 ? (
          <>
            {parsed.lines.map((l, i) => (
              <div
                key={i}
                className={`grid gap-2.5 ${l.hit ? 'text-hard' : ''}`}
                style={{ gridTemplateColumns: '32px 1fr' }}
              >
                <span className="text-right text-text-3">{l.num}</span>
                <span>{l.code}</span>
              </div>
            ))}
            {parsed.pointer && (
              <div className="grid gap-2.5 text-hard" style={{ gridTemplateColumns: '32px 1fr' }}>
                <span />
                <span>{parsed.pointer}</span>
              </div>
            )}
            <div
              className="text-hard font-medium mt-2 pt-2"
              style={{ borderTop: '1px solid var(--line)' }}
            >
              {parsed.message}
            </div>
          </>
        ) : (
          <div className="text-hard font-medium whitespace-pre-wrap">{errorText}</div>
        )}
      </div>

      {/* Hints list */}
      <h4 className="font-mono text-[10.5px] tracking-[.12em] uppercase text-text-3 font-medium mt-4 mb-2">
        {t('feedbackWhatToCheck')}
      </h4>
      <div className="rounded-[9px] border border-[var(--line)] overflow-hidden" style={{ background: 'var(--bg-elev)' }}>
        <ul className="list-none p-0 m-0">
          {getCompileHints(errorText).map((hintKey, i) => (
            <li
              key={i}
              className="flex items-start gap-2.5 py-2 px-3 text-[13px] text-text-2"
              style={{ borderTop: i > 0 ? '1px solid var(--line)' : undefined }}
            >
              <span
                className="w-[18px] h-[18px] rounded-[5px] flex items-center justify-center flex-shrink-0 font-mono text-[10px] text-text-3"
                style={{ background: 'var(--bg-sunken)', border: '1px solid var(--line)' }}
              >
                {i + 1}
              </span>
              <span>{t(hintKey)}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

function getLikelyCauses(error: string): TranslationKey[] {
  const e = error.toLowerCase();
  if (e.includes('shape') || e.includes('size') || e.includes('dimension'))
    return ['feedbackCauseShape1', 'feedbackCauseShape2', 'feedbackCauseShape3'];
  if (e.includes('nan') || e.includes('inf'))
    return ['feedbackCauseNan1', 'feedbackCauseNan2', 'feedbackCauseNan3'];
  if (e.includes('type') || e.includes('dtype'))
    return ['feedbackCauseDtype1', 'feedbackCauseDtype2'];
  return ['feedbackCauseDefault1', 'feedbackCauseDefault2'];
}

function getCompileHints(error: string): TranslationKey[] {
  const e = error.toLowerCase();
  if (e.includes('indent'))
    return ['feedbackCheckIndent1', 'feedbackCheckIndent2'];
  if (e.includes('top level'))
    return ['feedbackCheckTopLevel1', 'feedbackCheckTopLevel2'];
  return ['feedbackCheckSyntax1', 'feedbackCheckSyntax2'];
}

export function FeedbackModal({ result, onClose, attemptNumber, pathProgress, onViewHint }: FeedbackModalProps) {
  const { t } = useLocale();
  const modalRef = useRef<HTMLDivElement>(null);
  const kind = classifyResult(result);
  const config = kindConfig[kind];
  const Icon = config.icon;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const tag = `${t(config.tagKey)}${attemptNumber ? ` · #${attemptNumber}` : ''}`;

  return (
    <>
      {/* Scrim */}
      <div
        className="fixed inset-0 z-[90] animate-[fadeIn_.2s_ease]"
        style={{
          background: 'color-mix(in oklab, var(--text) 28%, transparent)',
          backdropFilter: 'saturate(180%) blur(4px)',
          WebkitBackdropFilter: 'saturate(180%) blur(4px)',
        }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        className="fixed top-1/2 left-1/2 z-[100] flex flex-col overflow-hidden animate-[modalIn_.2s_cubic-bezier(.16,1,.3,1)]"
        style={{
          transform: 'translate(-50%, -50%)',
          width: 'min(560px, calc(100vw - 40px))',
          maxHeight: '86vh',
          background: 'var(--bg-elev)',
          border: '1px solid var(--line)',
          borderRadius: '16px',
          boxShadow: '0 24px 64px -12px color-mix(in oklab, var(--text) 35%, transparent), 0 4px 12px -2px color-mix(in oklab, var(--text) 15%, transparent)',
        }}
      >
        {/* Head */}
        <div
          className={`flex-shrink-0 px-6 pt-[22px] pb-[18px] relative ${config.headClass}`}
          style={{ borderBottom: '1px solid var(--line)' }}
        >
          <div className="flex items-center gap-3.5">
            <div className={`w-[38px] h-[38px] rounded-[10px] inline-flex items-center justify-center flex-shrink-0 ${config.iconClass}`}>
              <Icon className="w-[18px] h-[18px]" strokeWidth={2.5} />
            </div>
            <div>
              <div className="font-mono text-[10.5px] tracking-[.12em] uppercase text-text-3">{tag}</div>
              <div className={`text-xl tracking-tight font-semibold mt-0.5 ${config.titleColor}`}>
                {t(config.titleKey)}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-7 h-7 rounded-[7px] inline-flex items-center justify-center text-text-3 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_6%,transparent)] transition-colors cursor-pointer"
            style={{ border: 'none', background: 'transparent' }}
            aria-label={t('feedbackClose')}
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 overflow-auto flex-1 min-h-0">
          {kind === 'accepted' && <AcceptedBody result={result} pathProgress={pathProgress} />}
          {kind === 'wrong' && <WrongBody result={result} onViewHint={onViewHint} />}
          {kind === 'runtime' && <RuntimeBody result={result} />}
          {kind === 'compile' && <CompileBody result={result} />}
        </div>

        {/* Foot */}
        <div
          className="flex-shrink-0 px-6 py-3.5 flex items-center gap-3 justify-end"
          style={{ borderTop: '1px solid var(--line)' }}
        >
          {attemptNumber && (
            <span className="mr-auto font-mono text-[11px] text-text-3 tracking-wider">
              {t('submissionAttempt', { n: attemptNumber })}
            </span>
          )}
          {kind === 'accepted' ? (
            <button
              onClick={onClose}
              className="h-9 px-3.5 text-[13.5px] font-medium rounded-[9px] cursor-pointer inline-flex items-center gap-2 transition-[transform,background] duration-150"
              style={{ background: 'var(--text)', color: 'var(--bg)', border: '1px solid var(--text)' }}
            >
              {t('continueBtn')}
            </button>
          ) : (
            <>
              <button
                onClick={onClose}
                className="h-9 px-3.5 text-[13.5px] font-medium rounded-[9px] cursor-pointer inline-flex items-center gap-2 text-text-2 hover:text-text transition-colors"
                style={{ background: 'transparent', border: 'none' }}
              >
                {t('feedbackClose')}
              </button>
              <button
                onClick={onClose}
                className="h-9 px-3.5 text-[13.5px] font-medium rounded-[9px] cursor-pointer inline-flex items-center gap-2 transition-[transform,background] duration-150"
                style={{ background: 'var(--text)', color: 'var(--bg)', border: '1px solid var(--text)' }}
              >
                {t('feedbackTryAgain')}
              </button>
            </>
          )}
        </div>
      </div>
    </>
  );
}
