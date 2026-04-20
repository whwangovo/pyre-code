'use client';

import Link from 'next/link';
import { StatusIcon } from './StatusIcon';
import { DifficultyBadge } from './DifficultyBadge';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, ProblemProgress } from '@/lib/types';

interface ProblemRowProps {
  problem: Problem;
  index: number;
  progress?: ProblemProgress;
}

export function ProblemRow({ problem, index, progress }: ProblemRowProps) {
  const status = progress?.status || 'todo';
  const { locale } = useLocale();

  return (
    <Link
      href={`/problems/${problem.id}`}
      className="flex items-center gap-4 px-4 py-3.5 rounded-xl hover:bg-surface-secondary transition-colors duration-200 group"
    >
      <StatusIcon status={status} />
      <span className="text-sm text-text-tertiary w-8">{index + 1}</span>
      <span className="flex-1 text-sm text-text-primary group-hover:text-accent transition-colors">
        {locale === 'zh' ? problem.titleZh : problem.title}
      </span>
      <DifficultyBadge difficulty={problem.difficulty} />
    </Link>
  );
}
