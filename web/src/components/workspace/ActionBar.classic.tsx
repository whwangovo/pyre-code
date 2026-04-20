'use client';

import { Play, Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/Button.classic';
import { GlassBar } from '@/components/ui/GlassBar.classic';
import { useLocale } from '@/context/LocaleContext';

interface ActionBarProps {
  onSubmit: () => void;
  onRun: () => void;
  isSubmitting: boolean;
  isRunning: boolean;
}

export function ActionBar({ onSubmit, onRun, isSubmitting, isRunning }: ActionBarProps) {
  const { t } = useLocale();
  const busy = isSubmitting || isRunning;
  return (
    <GlassBar className="px-4 py-2.5 flex items-center justify-end gap-3 border-t border-border/50">
      <Button
        variant="secondary"
        size="sm"
        onClick={onRun}
        disabled={busy}
      >
        {isRunning ? (
          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
        ) : (
          <Play className="w-3.5 h-3.5 mr-1.5" />
        )}
        {isRunning ? t('running') : t('run')}
      </Button>
      <Button
        variant="primary"
        size="sm"
        onClick={onSubmit}
        disabled={busy}
      >
        {isSubmitting ? (
          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
        ) : (
          <Send className="w-3.5 h-3.5 mr-1.5" />
        )}
        {isSubmitting ? t('judging') : t('submit')}
      </Button>
    </GlassBar>
  );
}
