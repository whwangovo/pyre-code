'use client';

import { Play, Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';

interface ActionBarProps {
  onSubmit: () => void;
  onRun: () => void;
  isSubmitting: boolean;
  isRunning: boolean;
  attemptCount?: number;
}

export function ActionBar({ onSubmit, onRun, isSubmitting, isRunning, attemptCount }: ActionBarProps) {
  const { t } = useLocale();
  const busy = isSubmitting || isRunning;
  return (
    <div
      className="px-4 py-2.5 flex items-center gap-3 flex-shrink-0"
      style={{
        borderTop: '1px solid var(--line)',
        backdropFilter: 'saturate(180%) blur(14px)',
        background: 'color-mix(in oklab, var(--bg) 82%, transparent)',
      }}
    >
      {attemptCount != null && attemptCount > 0 && (
        <span className="mono text-[11px] text-text-3">attempt #{attemptCount}</span>
      )}
      <div className="flex-1" />
      <Button variant="secondary" size="sm" onClick={onRun} disabled={busy}>
        {isRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
        {isRunning ? t('running') : t('run')}
        <span className="kbd">⌘↵</span>
      </Button>
      <Button variant="primary" size="sm" onClick={onSubmit} disabled={busy}>
        {isSubmitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
        {isSubmitting ? t('judging') : t('submit')}
        <span className="kbd">⌘⇧↵</span>
      </Button>
    </div>
  );
}
