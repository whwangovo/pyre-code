'use client';

import { useEffect, useState } from 'react';
import { TopNav } from '@/components/layout/TopNav.classic';
import { PathCard } from '@/components/path/PathCard';
import { useLocale } from '@/context/LocaleContext';
import type { LearningPath } from '@/lib/types';

type PathWithProgress = LearningPath & { solved: number; total: number };

export function PathsPageClassic() {
  const { t } = useLocale();
  const [paths, setPaths] = useState<PathWithProgress[]>([]);

  useEffect(() => {
    fetch('/api/paths')
      .then((r) => r.json())
      .then((d) => setPaths(d.paths ?? []));
  }, []);

  return (
    <div className="min-h-screen bg-surface">
      <TopNav />
      <main className="max-w-3xl mx-auto px-6 py-12">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-text-primary mb-2">{t('pathsHero')}</h1>
          <p className="text-sm text-text-secondary">{t('pathsSubtitle')}</p>
        </div>
        <div className="grid gap-3">
          {paths.map((path) => (
            <PathCard key={path.id} path={path} />
          ))}
        </div>
      </main>
    </div>
  );
}
