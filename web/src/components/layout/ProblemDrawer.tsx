'use client';

import Link from 'next/link';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/Badge';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, ProgressMap } from '@/lib/types';

interface ProblemDrawerProps {
  open: boolean;
  onClose: () => void;
  problems: Problem[];
  progress: ProgressMap;
  currentId?: string;
}

export function ProblemDrawer({ open, onClose, problems, progress, currentId }: ProblemDrawerProps) {
  const { t, tProblem } = useLocale();
  return (
    <>
      <div
        className={cn(
          'fixed inset-0 bg-black/20 z-40 transition-opacity duration-250',
          open ? 'opacity-100' : 'opacity-0 pointer-events-none'
        )}
        onClick={onClose}
      />
      <div
        className={cn(
          'fixed left-0 top-0 bottom-0 w-80 z-50 shadow-lg',
          'transform transition-transform duration-250 ease-out',
          open ? 'translate-x-0' : '-translate-x-full'
        )}
        style={{ background: 'var(--bg-elev)' }}
      >
        <div className="flex items-center justify-between p-4" style={{ borderBottom: '1px solid var(--line)' }}>
          <span className="font-semibold text-sm tracking-tight">{t('problems')}</span>
          <button onClick={onClose} className="p-1 rounded-lg text-text-2 hover:text-text transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="overflow-y-auto h-[calc(100%-57px)]">
          {problems.map((p) => {
            const status = progress[p.id]?.status || 'todo';
            return (
              <Link
                key={p.id}
                href={`/problems/${p.id}`}
                onClick={onClose}
                className={cn(
                  'flex items-center gap-3 px-4 py-3 text-sm transition-colors',
                  currentId === p.id
                    ? 'text-accent'
                    : 'text-text hover:bg-[color-mix(in_oklab,var(--text)_3%,transparent)]'
                )}
                style={currentId === p.id ? { background: 'var(--accent-wash)' } : undefined}
              >
                <span className={cn(
                  'w-2 h-2 rounded-full flex-shrink-0',
                )} style={{
                  background: status === 'solved' ? 'var(--easy)' : status === 'attempted' ? 'var(--medium)' : 'var(--line-strong)',
                }} />
                <span className="truncate flex-1">{tProblem(p.id)}</span>
                <Badge variant={p.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
                  {p.difficulty.toUpperCase()}
                </Badge>
              </Link>
            );
          })}
        </div>
      </div>
    </>
  );
}
