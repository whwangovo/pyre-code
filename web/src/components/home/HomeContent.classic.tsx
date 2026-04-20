'use client';

import Link from 'next/link';
import { ArrowRight, Zap, CheckCircle, FlaskConical, BarChart3, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/Button.classic';
import { useLocale } from '@/context/LocaleContext';

interface HomeContentProps {
  stats: { total: number; easy: number; medium: number; hard: number };
}

export function HomeContentClassic({ stats }: HomeContentProps) {
  const { t } = useLocale();

  const features = [
    { icon: CheckCircle, color: 'text-easy', bg: 'bg-easy/8', titleKey: 'feat1Title' as const, descKey: 'feat1Desc' as const },
    { icon: FlaskConical, color: 'text-accent', bg: 'bg-accent/8', titleKey: 'feat2Title' as const, descKey: 'feat2Desc' as const },
    { icon: BarChart3, color: 'text-medium', bg: 'bg-medium/8', titleKey: 'feat3Title' as const, descKey: 'feat3Desc' as const },
  ];

  const diffStats = [
    { key: 'Easy' as const, count: stats.easy, color: 'text-easy', bg: 'bg-easy/6', border: 'border-easy/20' },
    { key: 'Medium' as const, count: stats.medium, color: 'text-medium', bg: 'bg-medium/6', border: 'border-medium/20' },
    { key: 'Hard' as const, count: stats.hard, color: 'text-hard', bg: 'bg-hard/6', border: 'border-hard/20' },
  ];

  return (
    <main className="max-w-4xl mx-auto px-6">

      {/* Hero */}
      <section className="pt-20 pb-14 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-pill bg-accent/5 text-accent text-xs font-medium mb-5">
          <Zap className="w-3 h-3" />
          {t('heroPill', { count: stats.total })}
        </div>
        <h1 className="text-[3.25rem] font-bold tracking-tight text-text-primary mb-4 leading-[1.15] whitespace-pre-line">
          {t('heroTitle')}
        </h1>
        <p className="text-base text-text-secondary max-w-lg mx-auto mb-8 leading-relaxed whitespace-pre-line">
          {t('heroSubtitle')}
        </p>
        <div className="flex items-center justify-center gap-3">
          <Link href="/problems">
            <Button size="lg">
              {t('startPracticing')}
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </Link>
          <Link href="/paths">
            <Button size="lg" variant="secondary">
              <BookOpen className="w-4 h-4 mr-2" />
              {t('browsePaths')}
            </Button>
          </Link>
        </div>
      </section>

      {/* Feature cards */}
      <section className="grid grid-cols-3 gap-4 pb-8">
        {features.map(({ icon: Icon, color, bg, titleKey, descKey }) => (
          <div key={titleKey} className="rounded-2xl border border-border p-5 bg-surface hover:shadow-soft transition-shadow">
            <div className={`w-9 h-9 rounded-xl ${bg} flex items-center justify-center mb-3`}>
              <Icon className={`w-4.5 h-4.5 ${color}`} strokeWidth={2} />
            </div>
            <p className="text-sm font-semibold text-text-primary mb-1.5">{t(titleKey)}</p>
            <p className="text-xs text-text-secondary leading-relaxed">{t(descKey)}</p>
          </div>
        ))}
      </section>

      {/* Difficulty stats */}
      <section className="grid grid-cols-3 gap-4 pb-20">
        {diffStats.map((s) => (
          <div key={s.key} className={`${s.bg} border ${s.border} rounded-2xl p-5 text-center`}>
            <p className={`text-3xl font-bold tracking-tight ${s.color}`}>{s.count}</p>
            <p className="text-xs text-text-secondary mt-1 font-medium">{t(s.key)}</p>
          </div>
        ))}
      </section>

    </main>
  );
}
