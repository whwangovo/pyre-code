'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Layers, Eye, BarChart3, Zap, Target, TrendingUp } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav.classic';
import { PathStepList } from '@/components/path/PathStepList';
import { useLocale } from '@/context/LocaleContext';
import type { LearningPath } from '@/lib/types';

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  Layers, Eye, BarChart3, Zap, Target, TrendingUp,
};

interface PathStep {
  id: string;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  status: 'todo' | 'attempted' | 'solved';
}

type PathDetail = LearningPath & {
  problems: PathStep[];
  solved: number;
  total: number;
};

export function PathDetailPageClassic() {
  const { id } = useParams<{ id: string }>();
  const { locale, t } = useLocale();
  const [path, setPath] = useState<PathDetail | null>(null);

  useEffect(() => {
    fetch(`/api/paths/${id}`)
      .then((r) => r.json())
      .then((d) => setPath(d));
  }, [id]);

  if (!path) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <p className="text-sm text-text-tertiary">{t('loading')}</p>
      </div>
    );
  }

  const title = locale === 'zh' ? path.titleZh : path.titleEn;
  const description = locale === 'zh' ? path.descriptionZh : path.descriptionEn;
  const Icon = ICONS[path.icon] ?? Layers;
  const pct = path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;

  return (
    <div className="min-h-screen bg-surface">
      <TopNav />
      <main className="max-w-2xl mx-auto px-6 py-10">
        <Link
          href="/paths"
          className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary mb-6 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          {t('backToPaths')}
        </Link>

        <div className="flex items-start gap-4 mb-6">
          <div className="w-12 h-12 rounded-2xl bg-accent/8 flex items-center justify-center flex-shrink-0">
            <Icon className="w-6 h-6 text-accent" />
          </div>
          <div className="flex-1">
            <h1 className="text-xl font-bold text-text-primary mb-1">{title}</h1>
            <p className="text-sm text-text-secondary">{description}</p>
          </div>
        </div>

        <div className="flex items-center gap-3 mb-8">
          <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-sm text-text-secondary flex-shrink-0">
            {t('pathProgress', { solved: path.solved, total: path.total })}
          </span>
        </div>

        <PathStepList pathId={id} steps={path.problems} />
      </main>
    </div>
  );
}
