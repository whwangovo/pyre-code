'use client';

import { useEffect, useState } from 'react';
import { TopNav } from '@/components/layout/TopNav.classic';
import { ProblemFilters } from '@/components/problem/ProblemFilters';
import { ProblemRow } from '@/components/problem/ProblemRow';
import { useLocale } from '@/context/LocaleContext';
import type { Problem, ProgressMap } from '@/lib/types';

export function ProblemsPageClassic() {
  const { t } = useLocale();
  const [problems, setProblems] = useState<Problem[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [search, setSearch] = useState('');
  const [difficulty, setDifficulty] = useState('');
  const [status, setStatus] = useState('');

  useEffect(() => {
    fetch('/api/problems')
      .then((r) => r.json())
      .then((d) => setProblems(d.problems));
    fetch('/api/progress')
      .then((r) => r.json())
      .then((d) => setProgress(d.progress || {}));
  }, []);

  const solvedCount = Object.values(progress).filter((p) => p.status === 'solved').length;

  const filtered = problems.filter((p) => {
    if (search && !p.title.toLowerCase().includes(search.toLowerCase())) return false;
    if (difficulty && p.difficulty !== difficulty) return false;
    if (status) {
      const ps = progress[p.id]?.status || 'todo';
      if (ps !== status) return false;
    }
    return true;
  });

  return (
    <div className="min-h-screen bg-surface">
      <TopNav solvedCount={solvedCount} totalCount={problems.length} />
      <main className="max-w-4xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary mb-6">{t('problems')}</h1>
        <ProblemFilters
          search={search}
          onSearchChange={setSearch}
          difficulty={difficulty}
          onDifficultyChange={setDifficulty}
          status={status}
          onStatusChange={setStatus}
        />
        <div className="space-y-1">
          {filtered.map((p, i) => (
            <ProblemRow
              key={p.id}
              problem={p}
              index={i}
              progress={progress[p.id]}
            />
          ))}
          {filtered.length === 0 && (
            <p className="text-sm text-text-tertiary text-center py-12">{t('noMatch')}</p>
          )}
        </div>
      </main>
    </div>
  );
}
