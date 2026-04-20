'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useLocale } from '@/context/LocaleContext';
import { useDesign } from '@/context/DesignContext';
import { PathDetailPageClassic } from '@/components/paths/PathDetailPage.classic';
import { getProblemTitle } from '@/lib/i18n';
import { cn } from '@/lib/utils';
import type { LearningPath } from '@/lib/types';

interface PathStep {
  id: string;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  status: 'todo' | 'attempted' | 'solved';
}

type PathDetail = Omit<LearningPath, 'problems'> & {
  problems: PathStep[];
  solved: number;
  total: number;
};

export default function PathDetailPage() {
  const { design } = useDesign();
  if (design === 'classic') return <PathDetailPageClassic />;
  return <PathDetailPageNew />;
}

function PathDetailPageNew() {
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
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <p className="text-sm text-text-3">{t('loading')}</p>
      </div>
    );
  }

  const title = locale === 'zh' ? path.titleZh : path.titleEn;
  const description = locale === 'zh' ? path.descriptionZh : path.descriptionEn;
  const pct = path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;
  const attempted = path.problems.filter((p) => p.status === 'attempted').length;
  const currentStep = path.problems.find((p) => p.status === 'todo' || p.status === 'attempted');

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 pt-8 pb-20">
        {/* Breadcrumbs */}
        <div className="mono text-xs text-text-3 flex items-center gap-1.5 mb-5">
          <Link href="/paths" className="hover:text-text transition-colors">{t('paths')}</Link>
          <span className="opacity-60">/</span>
          <span className="text-text font-medium">{title}</span>
        </div>

        {/* Hero */}
        <div className="grid grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] max-[860px]:grid-cols-1 gap-12 pb-10 mb-10" style={{ borderBottom: '1px solid var(--line)' }}>
          <div>
            <div className="eyebrow mb-2">{t('paths')}</div>
            <h1 className="text-[clamp(32px,4vw,52px)] font-semibold tracking-[-0.03em] leading-[1.05] mb-3.5">{title}</h1>
            <p className="text-base text-text-2 leading-relaxed mb-5">{description}</p>
            <div className="flex gap-[22px] pt-5 flex-wrap" style={{ borderTop: '1px dashed var(--line)' }}>
              {[
                { k: t('problems'), v: String(path.total) },
                { k: t('Solved'), v: String(path.solved) },
                { k: t('Attempted'), v: String(attempted) },
              ].map((m) => (
                <div key={m.k}>
                  <div className="eyebrow">{m.k}</div>
                  <div className="mono text-sm mt-1 tabular-nums">{m.v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Progress card */}
          <div className="rounded-xl p-[22px]" style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}>
            <h4 className="mono text-[13px] tracking-[0.1em] uppercase text-text-3 font-medium mb-1">{t('progress')}</h4>
            <div className="text-[48px] font-semibold tracking-[-0.03em] tabular-nums leading-[1.1] flex items-baseline gap-2 mt-1.5 mb-4">
              {pct}<small className="text-lg text-text-3 font-medium">%</small>
            </div>
            <div className="h-1 rounded-pill relative mb-3.5" style={{ background: 'var(--line)' }}>
              <div className="absolute inset-0 rounded-pill" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
            </div>
            <div className="flex gap-3 mono text-xs text-text-2">
              <div className="flex-1 p-2.5 rounded-lg" style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}>
                <div className="text-text-3">{t('Solved')}</div>
                <div className="text-lg text-text tabular-nums mt-0.5">{path.solved}</div>
              </div>
              <div className="flex-1 p-2.5 rounded-lg" style={{ border: '1px solid var(--line)', background: 'var(--bg-sunken)' }}>
                <div className="text-text-3">{t('Attempted')}</div>
                <div className="text-lg text-text tabular-nums mt-0.5">{attempted}</div>
              </div>
            </div>
            {currentStep && (
              <div className="flex gap-2.5 mt-[18px]">
                <Link href={`/problems/${currentStep.id}?path=${id}`} className="flex-1">
                  <Button className="w-full">
                    {path.solved > 0 ? t('continueBtn') : t('startBtn')}
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Button>
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* Steps */}
        <div className="flex justify-between items-baseline mb-5">
          <h2 className="text-[22px] font-semibold tracking-[-0.02em]">{t('pathSteps')}</h2>
          <span className="mono text-xs text-text-3">{t('pathStepsNote')}</span>
        </div>

        <div className="relative pl-8">
          {/* Timeline line */}
          <div className="absolute left-[11px] top-0 bottom-0 w-px" style={{ background: 'var(--line)' }} />

          {path.problems.map((step, i) => {
            const stepTitle = getProblemTitle(step.id, locale);
            const isSolved = step.status === 'solved';
            const isAttempted = step.status === 'attempted';
            const isCurrent = !isSolved && !isAttempted && i === path.problems.findIndex((s) => s.status !== 'solved');

            return (
              <Link
                key={step.id}
                href={`/problems/${step.id}?path=${id}`}
                className={cn(
                  'group flex items-center gap-5 py-3.5 px-4 rounded-[10px] mb-1 transition-[background,border-color] duration-150',
                  isCurrent ? '' : 'hover:bg-[color-mix(in_oklab,var(--text)_3%,transparent)]'
                )}
                style={isCurrent ? { background: 'var(--accent-wash)', border: '1px solid var(--accent-line)' } : { border: '1px solid transparent' }}
              >
                {/* Step number */}
                <span
                  className={cn(
                    'w-6 h-6 rounded-full flex items-center justify-center mono text-[11px] flex-shrink-0 -ml-[42px]',
                    isSolved ? 'bg-easy text-white' : isAttempted ? 'text-medium' : isCurrent ? 'text-accent font-medium' : 'text-text-3'
                  )}
                  style={
                    isSolved
                      ? undefined
                      : isAttempted
                      ? { border: '1px solid var(--medium)', background: 'var(--bg-elev)' }
                      : isCurrent
                      ? { border: '1px solid var(--accent)', background: 'var(--accent-wash)' }
                      : { border: '1px solid var(--line)', background: 'var(--bg-sunken)' }
                  }
                >
                  {isSolved ? (
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                  ) : (
                    String(i + 1).padStart(2, '0')
                  )}
                </span>

                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm">{stepTitle}</div>
                  <div className="mono text-[11.5px] text-text-3 mt-0.5">{step.id}.py</div>
                </div>

                <Badge variant={step.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
                  {step.difficulty.toUpperCase()}
                </Badge>

                <span
                  className="mono text-[11px] px-[7px] py-[2px] rounded-[5px] tracking-[0.04em]"
                  style={
                    isSolved
                      ? { color: 'var(--easy)', border: '1px solid color-mix(in oklab, var(--easy) 30%, var(--line))', background: 'color-mix(in oklab, var(--easy) 8%, var(--bg-elev))' }
                      : isAttempted
                      ? { color: 'var(--medium)', border: '1px solid color-mix(in oklab, var(--medium) 30%, var(--line))', background: 'color-mix(in oklab, var(--medium) 8%, var(--bg-elev))' }
                      : isCurrent
                      ? { color: 'var(--accent)', border: '1px solid var(--accent-line)', background: 'var(--accent-wash)' }
                      : { color: 'var(--text-3)', border: '1px solid var(--line)', background: 'var(--bg-sunken)' }
                  }
                >
                  {isSolved ? 'SOLVED' : isAttempted ? 'ATTEMPTED' : isCurrent ? 'CURRENT' : 'TODO'}
                </span>

                <ArrowRight className="w-4 h-4 text-text-3 group-hover:text-accent transition-colors flex-shrink-0" />
              </Link>
            );
          })}
        </div>
      </main>
      <Footer />
    </div>
  );
}
