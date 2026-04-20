'use client';

import { useEffect, useRef, useState } from 'react';
import * as Collapsible from '@radix-ui/react-collapsible';
import { ChevronDown, Loader2, Settings2, Sparkles, ServerCog } from 'lucide-react';
import { Button } from '@/components/ui/Button.classic';
import { useLocale } from '@/context/LocaleContext';
import { MarkdownContent } from '@/components/workspace/MarkdownContent.classic';
import { getSampleTests, getSolutionCode } from '@/lib/problemContext';
import { useProblemStore } from '@/store/problemStore';
import type { Problem, AiHelpRequest, AiHelpResponse } from '@/lib/types';

const STORAGE_KEY = 'ai_help_config';
const CUSTOM_PROMPT_STORAGE_KEY = 'ai_help_custom_prompt';

interface AIHelpTabProps {
  problem: Problem;
}

export function AIHelpTab({ problem }: AIHelpTabProps) {
  const { locale, t } = useLocale();
  const hasLoadedConfig = useRef(false);
  const [serverConfigured, setServerConfigured] = useState(false);
  const {
    currentCode,
    aiHelpConfig,
    setAiHelpConfig,
    aiHelpConfigOpen,
    setAiHelpConfigOpen,
    aiHelpCustomPrompt,
    setAiHelpCustomPrompt,
    aiHelpResponse,
    setAiHelpResponse,
    aiHelpError,
    setAiHelpError,
    aiHelpLoading,
    setAiHelpLoading,
  } = useProblemStore();

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
      });
    } catch {
      // Ignore malformed local config.
    }

    const savedPrompt = localStorage.getItem(CUSTOM_PROMPT_STORAGE_KEY);
    if (savedPrompt !== null) {
      setAiHelpCustomPrompt(savedPrompt);
    }
  }, [setAiHelpConfig, setAiHelpCustomPrompt]);

  useEffect(() => {
    if (!hasLoadedConfig.current) return;
    const { baseUrl, apiKey, model } = aiHelpConfig;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ baseUrl, apiKey, model }));
  }, [aiHelpConfig.baseUrl, aiHelpConfig.apiKey, aiHelpConfig.model]);

  useEffect(() => {
    if (!hasLoadedConfig.current) return;
    localStorage.setItem(CUSTOM_PROMPT_STORAGE_KEY, aiHelpCustomPrompt);
  }, [aiHelpCustomPrompt]);

  useEffect(() => {
    fetch('/api/ai-help/status')
      .then((r) => r.json())
      .then((d: { configured: boolean }) => setServerConfigured(d.configured))
      .catch(() => setServerConfigured(false));
  }, []);

  const handleGenerate = async () => {
    if (!serverConfigured && (!aiHelpConfig.baseUrl.trim() || !aiHelpConfig.apiKey.trim() || !aiHelpConfig.model.trim())) {
      setAiHelpConfigOpen(true);
      setAiHelpError(t('aiHelpMissingConfig'));
      setAiHelpResponse(null);
      return;
    }

    setAiHelpLoading(true);
    setAiHelpError(null);
    setAiHelpResponse(null);
    const trimmedCustomPrompt = aiHelpCustomPrompt.trim();

    const requestBody: AiHelpRequest = {
      problemId: problem.id,
      problemTitle: problem.title,
      functionName: problem.functionName,
      description: locale === 'zh' ? problem.descriptionZh : problem.descriptionEn,
      solutionCode: trimmedCustomPrompt ? '' : getSolutionCode(problem.id),
      sampleTests: getSampleTests(problem),
      locale,
      config: {
        baseUrl: aiHelpConfig.baseUrl.trim(),
        apiKey: aiHelpConfig.apiKey.trim(),
        model: aiHelpConfig.model.trim(),
      },
      ...(trimmedCustomPrompt ? { customPrompt: trimmedCustomPrompt } : {}),
      ...(aiHelpConfig.includeUserCode && currentCode.trim() ? { userCode: currentCode } : {}),
    };

    try {
      const response = await fetch('/api/ai-help', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      if (!response.ok) {
        setAiHelpResponse(null);
        setAiHelpError(data.error || t('aiHelpRequestFailed'));
        return;
      }

      const result = data as AiHelpResponse;
      setAiHelpResponse(result.guidance);
    } catch {
      setAiHelpResponse(null);
      setAiHelpError(t('aiHelpRequestFailed'));
    } finally {
      setAiHelpLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="rounded-xl border border-border/60 bg-surface-secondary/60 p-4 space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
            <Sparkles className="w-4 h-4 text-accent" />
            <span>{t('aiHelp')}</span>
          </div>
          <p className="text-xs text-text-secondary">{t('aiHelpSafetyNote')}</p>
        </div>

        <label className="block space-y-1">
          <span className="text-xs font-medium text-text-secondary">{t('aiHelpCustomPrompt')}</span>
          <textarea
            value={aiHelpCustomPrompt}
            onChange={(e) => setAiHelpCustomPrompt(e.target.value)}
            className="min-h-[96px] w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent/50"
            placeholder={t('aiHelpCustomPromptPlaceholder')}
          />
        </label>

        <div className="flex items-center gap-3">
          <label className="flex flex-1 items-center gap-2 text-sm text-text-secondary">
            <input
              type="checkbox"
              checked={aiHelpConfig.includeUserCode}
              onChange={(e) => setAiHelpConfig({ includeUserCode: e.target.checked })}
              className="rounded border-border text-accent focus:ring-accent"
            />
            <span>{t('aiHelpIncludeCode')}</span>
          </label>
          <Button
            variant="primary"
            size="sm"
            onClick={handleGenerate}
            disabled={aiHelpLoading}
            className="shrink-0"
          >
            {aiHelpLoading ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Sparkles className="mr-1.5 h-3.5 w-3.5" />}
            {aiHelpLoading ? t('aiHelpGenerating') : t('aiHelpGenerate')}
          </Button>
        </div>

        <Collapsible.Root open={aiHelpConfigOpen} onOpenChange={setAiHelpConfigOpen}>
          <Collapsible.Trigger asChild>
            <button className="flex w-full items-center justify-between rounded-lg border border-border/60 bg-white px-3 py-2 text-sm text-text-secondary transition-colors hover:text-text-primary">
              <span className="flex items-center gap-2">
                <Settings2 className="h-4 w-4" />
                {aiHelpConfigOpen ? t('aiHelpHideConfig') : t('aiHelpShowConfig')}
              </span>
              <ChevronDown className={`h-4 w-4 transition-transform ${aiHelpConfigOpen ? 'rotate-180' : ''}`} />
            </button>
          </Collapsible.Trigger>

          <Collapsible.Content className="space-y-3 pt-3">
            {serverConfigured ? (
              <div className="flex items-center gap-2 rounded-lg border border-easy/30 bg-easy/5 px-3 py-2 text-sm text-easy">
                <ServerCog className="h-4 w-4" />
                {t('aiHelpServerConfigured')}
              </div>
            ) : (
              <>
                <div className="text-xs font-medium text-text-secondary">{t('aiHelpSetup')}</div>

            <label className="block space-y-1">
              <span className="text-xs font-medium text-text-secondary">{t('aiHelpBaseUrl')}</span>
              <input
                type="text"
                value={aiHelpConfig.baseUrl}
                onChange={(e) => setAiHelpConfig({ baseUrl: e.target.value })}
                className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent/50"
                placeholder="https://api.openai.com/v1"
              />
            </label>

            <label className="block space-y-1">
              <span className="text-xs font-medium text-text-secondary">{t('aiHelpApiKey')}</span>
              <input
                type="password"
                value={aiHelpConfig.apiKey}
                onChange={(e) => setAiHelpConfig({ apiKey: e.target.value })}
                className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent/50"
                placeholder="sk-..."
              />
            </label>

            <label className="block space-y-1">
              <span className="text-xs font-medium text-text-secondary">{t('aiHelpModel')}</span>
              <input
                type="text"
                value={aiHelpConfig.model}
                onChange={(e) => setAiHelpConfig({ model: e.target.value })}
                className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent/50"
                placeholder="gpt-4o-mini"
              />
            </label>
              </>
            )}
          </Collapsible.Content>
        </Collapsible.Root>
      </div>

      {aiHelpError && (
        <div className="rounded-xl border border-hard/20 bg-hard/5 px-4 py-3 text-sm text-hard whitespace-pre-wrap">
          {aiHelpError}
        </div>
      )}

      {aiHelpResponse ? (
        <div className="rounded-xl border border-border/60 bg-white px-4 py-4">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-text-tertiary">
            {t('aiHelpResponseTitle')}
          </div>
          <div className="text-sm text-text-secondary">
            <MarkdownContent content={aiHelpResponse} />
          </div>
        </div>
      ) : !aiHelpError && (
        <div className="rounded-xl border border-dashed border-border px-4 py-6 text-sm text-text-tertiary">
          {t('aiHelpEmpty')}
        </div>
      )}
    </div>
  );
}
