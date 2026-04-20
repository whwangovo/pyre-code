'use client';

import Link from 'next/link';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/Badge.classic';
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
          'fixed left-0 top-0 bottom-0 w-80 bg-surface z-50 shadow-soft-lg',
          'transform transition-transform duration-250 ease-out',
          open ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex items-center justify-between p-4 border-b border-border">
          <span className="font-semibold text-sm tracking-tight">{t('problems')}</span>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-gray-100 transition-colors">
            <X className="w-4 h-4 text-text-secondary" />
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
                    ? 'bg-accent/5 text-accent'
                    : 'text-text-primary hover:bg-gray-50'
                )}
              >
                <span className={cn(
                  'w-2 h-2 rounded-full flex-shrink-0',
                  status === 'solved' && 'bg-solved',
                  status === 'attempted' && 'bg-medium',
                  status === 'todo' && 'bg-gray-300',
                )} />
                <span className="truncate flex-1">{tProblem(p.id)}</span>
                <Badge variant={p.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
                  {t(p.difficulty as 'Easy' | 'Medium' | 'Hard')}
                </Badge>
              </Link>
            );
          })}
        </div>
      </div>
    </>
  );
}
