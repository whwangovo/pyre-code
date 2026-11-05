'use client';

import { useEffect, useRef, useState } from 'react';
import * as Collapsible from '@radix-ui/react-collapsible';
import { ArrowUp, Check, Loader2, RefreshCw, RotateCcw, Settings2, Sparkles, ServerCog, Square, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';
import { MarkdownContent } from '@/components/workspace/MarkdownContent';
import { AI_HELP_QUICK_ACTIONS, HOT100_QUICK_ACTIONS } from '@/lib/aiHelpPrompts';
import { saveAiChat } from '@/lib/aiChatHistory';
import { getSampleTests } from '@/lib/problemContext';
import { cn } from '@/lib/utils';
import { useProblemStore } from '@/store/problemStore';
import type { Problem, AiChatMessage, AiHelpRequest } from '@/lib/types';

const STORAGE_KEY = 'ai_help_config';
const INPUT_KEY_PREFIX = 'ai_help_input:';
const INPUT_MAX_HEIGHT = 160;

// The unsent message is kept per problem, since switching tabs or problems unmounts this panel
function loadUnsentInput(problemId: string): string {
  try {
    return localStorage.getItem(INPUT_KEY_PREFIX + problemId) ?? '';
  } catch {
    return '';
  }
}

function saveUnsentInput(problemId: string, value: string) {
  try {
    if (value) localStorage.setItem(INPUT_KEY_PREFIX + problemId, value);
    else localStorage.removeItem(INPUT_KEY_PREFIX + problemId);
  } catch {
    // Ignore storage write failures.
  }
}

const inputStyle = {
  background: 'var(--bg-elev)',
  border: '1px solid var(--line)',
};

const iconButtonClass =
  'inline-flex items-center gap-1.5 h-7 px-2 rounded-md text-xs text-text-3 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] transition-colors cursor-pointer disabled:opacity-40 disabled:pointer-events-none';

interface AIHelpTabProps {
  problem: Problem;
}

export function AIHelpTab({ problem }: AIHelpTabProps) {
  const { locale, t } = useLocale();
  const isHot100 = problem.track === 'hot100';
  const hasLoadedConfig = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [input, setInputState] = useState(() => loadUnsentInput(problem.id));
  const setInput = (value: string) => {
    setInputState(value);
    saveUnsentInput(problem.id, value);
  };
  const [serverConfigured, setServerConfigured] = useState(false);
  const aiHelpConfig = useProblemStore((s) => s.aiHelpConfig);
  const setAiHelpConfig = useProblemStore((s) => s.setAiHelpConfig);
  const aiHelpConfigOpen = useProblemStore((s) => s.aiHelpConfigOpen);
  const setAiHelpConfigOpen = useProblemStore((s) => s.setAiHelpConfigOpen);
  const aiHelpMessages = useProblemStore((s) => s.aiHelpMessages);
  const setAiHelpMessages = useProblemStore((s) => s.setAiHelpMessages);
  const aiHelpError = useProblemStore((s) => s.aiHelpError);
  const setAiHelpError = useProblemStore((s) => s.setAiHelpError);
  const aiHelpLoading = useProblemStore((s) => s.aiHelpLoading);
  const setAiHelpLoading = useProblemStore((s) => s.setAiHelpLoading);
  const setAiHelpAbortController = useProblemStore((s) => s.setAiHelpAbortController);
  const resetAiHelp = useProblemStore((s) => s.resetAiHelp);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    hasLoadedConfig.current = true;
    if (!saved) return;
    try {
      const parsed = JSON.parse(saved) as Partial<typeof aiHelpConfig>;
      setAiHelpConfig({
        baseUrl: parsed.baseUrl ?? '',
        apiKey: parsed.apiKey ?? '',
        model: parsed.model ?? '',
        ...(typeof parsed.includeUserCode === 'boolean' && { includeUserCode: parsed.includeUserCode }),
      });
    } catch {}
  }, [setAiHelpConfig]);

  // Read from the store, not the render closure: on mount the closure still holds the
  // pre-load values, and persisting those would clobber what was just loaded.
  useEffect(() => {
    if (!hasLoadedConfig.current) return;
    const { baseUrl, apiKey, model, includeUserCode } = useProblemStore.getState().aiHelpConfig;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ baseUrl, apiKey, model, includeUserCode }));
  }, [aiHelpConfig.baseUrl, aiHelpConfig.apiKey, aiHelpConfig.model, aiHelpConfig.includeUserCode]);

  useEffect(() => {
    fetch('/api/ai-help/status')
      .then((r) => r.json())
      .then((d: { configured: boolean }) => setServerConfigured(d.configured))
      .catch(() => setServerConfigured(false));
  }, []);

  // Follow streaming output, unless the learner has scrolled up to read something
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [aiHelpMessages, aiHelpError]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px`;
  }, [input]);

  const requestReply = async (history: AiChatMessage[]) => {
    const { aiHelpConfig: config, currentCode } = useProblemStore.getState();
    if (!serverConfigured && (!config.baseUrl.trim() || !config.apiKey.trim() || !config.model.trim())) {
      setAiHelpMessages(history);
      saveAiChat(problem.id, history);
      setAiHelpError(t('aiHelpMissingConfig'));
      setAiHelpConfigOpen(true);
      return;
    }

    const controller = new AbortController();
    const isCurrent = () => useProblemStore.getState().aiHelpAbortController === controller;
    setAiHelpAbortController(controller);
    setAiHelpLoading(true);
    setAiHelpError(null);
    setAiHelpMessages([...history, { role: 'assistant', content: '' }]);
    // Saved now too, so the question survives leaving before the reply finishes
    saveAiChat(problem.id, history);
    stickToBottom.current = true;

    try {
      const { codeMode } = useProblemStore.getState();
      const isAcm = isHot100 && codeMode === 'acm' && problem.acm;
      const description = locale === 'zh' ? problem.descriptionZh : problem.descriptionEn;
      const body: AiHelpRequest = {
        problemId: problem.id,
        problemTitle: problem.title,
        functionName: problem.functionName,
        description: isAcm
          ? `${description}\n\nInput format:\n${problem.acm!.inputEn}\n\nOutput format:\n${problem.acm!.outputEn}`
          : description,
        sampleTests: getSampleTests(problem),
        messages: history.map(({ role, content }) => ({ role, content })),
        userCode: config.includeUserCode !== false ? currentCode : undefined,
        track: isHot100 ? 'hot100' : undefined,
        mode: isHot100 ? codeMode : undefined,
        locale,
        config: { baseUrl: config.baseUrl, apiKey: config.apiKey, model: config.model },
      };
      const res = await fetch('/api/ai-help', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.error ? `${t('aiHelpRequestFailed')} ${data.error}` : t('aiHelpRequestFailed'));
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (!chunk || !isCurrent()) continue;
        setAiHelpMessages((prev) => {
          const last = prev.at(-1);
          if (last?.role !== 'assistant') return prev;
          return [...prev.slice(0, -1), { ...last, content: last.content + chunk }];
        });
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setAiHelpError(err instanceof Error && err.message ? err.message : t('aiHelpRequestFailed'));
      }
    } finally {
      // A reset (e.g. switching problems) already cleared everything and swapped the controller
      if (isCurrent()) {
        setAiHelpMessages((prev) => {
          const last = prev.at(-1);
          return last?.role === 'assistant' && !last.content ? prev.slice(0, -1) : prev;
        });
        saveAiChat(problem.id, useProblemStore.getState().aiHelpMessages);
        setAiHelpLoading(false);
        setAiHelpAbortController(null);
      }
    }
  };

  const send = (content: string, label?: string) => {
    if (aiHelpLoading || !content.trim()) return;
    const history = [...useProblemStore.getState().aiHelpMessages, { role: 'user' as const, content: content.trim(), label }];
    if (!label) setInput('');
    requestReply(history);
  };

  const retry = () => {
    const history = useProblemStore.getState().aiHelpMessages;
    if (aiHelpLoading || history.at(-1)?.role !== 'user') return;
    requestReply(history);
  };

  const canRetry = !aiHelpLoading && aiHelpError !== null && aiHelpMessages.at(-1)?.role === 'user';

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-1 px-3 h-10 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
        <label className="flex items-center gap-2 text-xs text-text-2 mr-auto cursor-pointer">
          <input
            type="checkbox"
            checked={aiHelpConfig.includeUserCode !== false}
            onChange={(e) => setAiHelpConfig({ includeUserCode: e.target.checked })}
            className="accent-[var(--accent)]"
          />
          {t('aiHelpIncludeCode')}
        </label>
        <button
          type="button"
          onClick={() => setAiHelpConfigOpen(!aiHelpConfigOpen)}
          className={cn(iconButtonClass, aiHelpConfigOpen && 'text-text')}
          title={aiHelpConfigOpen ? t('aiHelpHideConfig') : t('aiHelpShowConfig')}
        >
          {serverConfigured ? <ServerCog className="w-3.5 h-3.5" /> : <Settings2 className="w-3.5 h-3.5" />}
          <span className="max-w-[160px] truncate">{(!serverConfigured && aiHelpConfig.model) || t('aiHelpSetup')}</span>
        </button>
        <button
          type="button"
          onClick={() => {
            resetAiHelp();
            saveAiChat(problem.id, []);
          }}
          disabled={aiHelpMessages.length === 0 && !aiHelpError}
          className={iconButtonClass}
          title={t('aiHelpClear')}
          aria-label={t('aiHelpClear')}
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <Collapsible.Root open={aiHelpConfigOpen} onOpenChange={setAiHelpConfigOpen}>
        <Collapsible.Content className="px-4 py-3 space-y-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--line)' }}>
          {serverConfigured ? (
            <div className="flex items-center gap-1.5 text-xs text-text-3">
              <ServerCog className="w-3.5 h-3.5" />
              {t('aiHelpServerConfigured')}
            </div>
          ) : (
            <>
              <label className="block space-y-1">
                <span className="text-xs font-medium text-text-2">{t('aiHelpBaseUrl')}</span>
                <input
                  type="text"
                  value={aiHelpConfig.baseUrl}
                  onChange={(e) => setAiHelpConfig({ baseUrl: e.target.value })}
                  className="w-full rounded-lg px-3 py-2 text-sm text-text outline-none"
                  style={inputStyle}
                  placeholder="https://api.openai.com/v1"
                />
              </label>
              <label className="block space-y-1">
                <span className="text-xs font-medium text-text-2">{t('aiHelpApiKey')}</span>
                <input
                  type="password"
                  value={aiHelpConfig.apiKey}
                  onChange={(e) => setAiHelpConfig({ apiKey: e.target.value })}
                  className="w-full rounded-lg px-3 py-2 text-sm text-text outline-none"
                  style={inputStyle}
                  placeholder="sk-..."
                />
              </label>
              <ModelField />
            </>
          )}
        </Collapsible.Content>
      </Collapsible.Root>

      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
        className="flex-1 min-h-0 overflow-y-auto px-4 py-4 space-y-4"
      >
        {aiHelpMessages.length === 0 && !aiHelpError && (
          <div className="h-full flex flex-col items-center justify-center gap-2 text-center px-6">
            <Sparkles className="w-5 h-5 text-text-3" />
            <p className="text-sm text-text-2">{t('aiHelpEmpty')}</p>
            <p className="text-xs text-text-3">{t('aiHelpSafetyNote')}</p>
          </div>
        )}

        {aiHelpMessages.map((message, index) =>
          message.role === 'user' ? (
            <div key={index} className="flex justify-end">
              <div
                className="max-w-[85%] rounded-2xl rounded-br-md px-3.5 py-2 text-sm text-text whitespace-pre-wrap break-words"
                style={{ background: 'var(--accent-wash)', border: '1px solid var(--accent-line)' }}
              >
                {message.label ? (
                  <span className="inline-flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-accent" />
                    {message.label}
                  </span>
                ) : message.content}
              </div>
            </div>
          ) : (
            <div key={index} className="text-sm text-text-2">
              {message.content ? (
                <MarkdownContent content={message.content} />
              ) : (
                <Loader2 className="w-4 h-4 animate-spin text-text-3" />
              )}
            </div>
          ),
        )}

        {aiHelpError && (
          <div
            className="rounded-xl px-4 py-3 text-sm text-hard whitespace-pre-wrap flex items-start justify-between gap-3"
            style={{ border: '1px solid color-mix(in oklab, var(--hard) 20%, var(--line))', background: 'color-mix(in oklab, var(--hard) 5%, var(--bg-elev))' }}
          >
            <span>{aiHelpError}</span>
            {canRetry && (
              <button type="button" onClick={retry} className="inline-flex items-center gap-1 text-xs shrink-0 hover:underline cursor-pointer">
                <RotateCcw className="w-3 h-3" />
                {t('aiHelpRetry')}
              </button>
            )}
          </div>
        )}
      </div>

      <div className="flex-shrink-0 px-3 pt-2 pb-3 space-y-2" style={{ borderTop: '1px solid var(--line)' }}>
        <div className="flex flex-wrap gap-1.5">
          {(isHot100 ? HOT100_QUICK_ACTIONS : AI_HELP_QUICK_ACTIONS).map((action) => (
            <button
              key={action.id}
              type="button"
              disabled={aiHelpLoading}
              onClick={() => send(action.prompt[locale], action.label[locale])}
              className="rounded-full border border-line px-2.5 py-0.5 text-xs text-text-2 hover:border-line-strong hover:text-text transition-colors cursor-pointer disabled:opacity-50 disabled:pointer-events-none"
            >
              {action.label[locale]}
            </button>
          ))}
        </div>
        <div className="flex items-end gap-2 rounded-xl px-3 py-2" style={inputStyle}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              // isComposing: Enter confirms an IME candidate (e.g. Chinese pinyin), don't send
              if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send(input);
              }
            }}
            rows={1}
            className="flex-1 min-w-0 bg-transparent text-sm text-text outline-none resize-none py-1"
            placeholder={t('aiHelpInputPlaceholder')}
            aria-label={t('aiHelpInputPlaceholder')}
          />
          {aiHelpLoading ? (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => useProblemStore.getState().aiHelpAbortController?.abort()}
              className="h-7 w-7 px-0 shrink-0"
              title={t('aiHelpStop')}
              aria-label={t('aiHelpStop')}
            >
              <Square className="w-3 h-3 fill-current" />
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={() => send(input)}
              disabled={!input.trim()}
              className="h-7 w-7 px-0 shrink-0"
              title={t('aiHelpSend')}
              aria-label={t('aiHelpSend')}
            >
              <ArrowUp className="w-3.5 h-3.5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

/** Model name input with a "fetch models" button that opens a filterable dropdown. */
function ModelField() {
  const { t } = useLocale();
  const aiHelpConfig = useProblemStore((s) => s.aiHelpConfig);
  const setAiHelpConfig = useProblemStore((s) => s.setAiHelpConfig);
  const boxRef = useRef<HTMLDivElement>(null);
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    if (!listOpen) return;
    const handlePointerDown = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setListOpen(false);
    };
    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, [listOpen]);

  const handleFetch = async () => {
    if (!aiHelpConfig.baseUrl.trim() || !aiHelpConfig.apiKey.trim()) {
      setError(t('aiHelpFetchModelsNeedConfig'));
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/ai-help/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseUrl: aiHelpConfig.baseUrl, apiKey: aiHelpConfig.apiKey }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data: { models: string[] } = await res.json();
      setModels(data.models);
      setQuery('');
      setListOpen(true);
    } catch {
      setError(t('aiHelpFetchModelsFailed'));
    } finally {
      setLoading(false);
    }
  };

  const filtered = models.filter((m) => m.toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <div className="space-y-1">
      <span className="text-xs font-medium text-text-2">{t('aiHelpModel')}</span>
      <div ref={boxRef} className="relative">
        <div className="flex gap-2">
          <input
            type="text"
            value={aiHelpConfig.model}
            onChange={(e) => {
              setAiHelpConfig({ model: e.target.value });
              setQuery(e.target.value);
              if (models.length) setListOpen(true);
            }}
            onFocus={() => models.length && setListOpen(true)}
            onKeyDown={(e) => e.key === 'Escape' && setListOpen(false)}
            className="w-full min-w-0 rounded-lg px-3 py-2 text-sm text-text outline-none"
            style={inputStyle}
            placeholder="gpt-4o-mini"
            aria-label={t('aiHelpModel')}
            autoComplete="off"
          />
          <Button variant="secondary" onClick={handleFetch} disabled={loading} className="shrink-0">
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
            {t('aiHelpFetchModels')}
          </Button>
        </div>
        {listOpen && models.length > 0 && (
          <ul
            className="absolute z-20 mt-1 w-full max-h-60 overflow-y-auto rounded-lg py-1 text-sm shadow-lg"
            style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
          >
            {filtered.length ? filtered.map((m) => (
              <li key={m}>
                <button
                  type="button"
                  onClick={() => {
                    setAiHelpConfig({ model: m });
                    setListOpen(false);
                  }}
                  className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-text-2 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)] cursor-pointer"
                >
                  <span className="truncate">{m}</span>
                  {m === aiHelpConfig.model && <Check className="w-3.5 h-3.5 shrink-0 text-accent" />}
                </button>
              </li>
            )) : (
              <li className="px-3 py-1.5 text-text-3">{t('aiHelpNoModels')}</li>
            )}
          </ul>
        )}
      </div>
      {error && <div className="text-xs text-hard">{error}</div>}
    </div>
  );
}
