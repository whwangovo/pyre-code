'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { useLocale } from '@/context/LocaleContext';
import { useDesign } from '@/context/DesignContext';
import { PathsPageClassic } from '@/components/paths/PathsPage.classic';
import type { LearningPath } from '@/lib/types';

type PathWithProgress = LearningPath & { solved: number; total: number };

export default function PathsPage() {
  const { design } = useDesign();
  if (design === 'classic') return <PathsPageClassic />;
  return <PathsPageNew />;
}

function PathsPageNew() {
  const { locale, t } = useLocale();
  const [paths, setPaths] = useState<PathWithProgress[]>([]);

  useEffect(() => {
    fetch('/api/paths')
      .then((r) => r.json())
      .then((d) => setPaths(d.paths ?? []));
  }, []);

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 pt-10 pb-20">
        <div className="mb-12 pb-7 max-w-[780px]" style={{ borderBottom: '1px solid var(--line)' }}>
          <div className="eyebrow mb-2.5">{t('paths')}</div>
          <h1 className="text-[clamp(32px,4vw,52px)] font-semibold tracking-[-0.032em] leading-[1.05] mb-3.5">
            {t('pathsHero')}
          </h1>
          <p className="text-base text-text-2 leading-relaxed max-w-[58ch]">{t('pathsSubtitle')}</p>
        </div>

        <div
          className="grid grid-cols-2 max-[820px]:grid-cols-1 rounded-[14px] overflow-hidden"
          style={{ gap: '1px', background: 'var(--line)', border: '1px solid var(--line)' }}
        >
          {paths.map((path, idx) => {
            const title = locale === 'zh' ? path.titleZh : path.titleEn;
            const desc = locale === 'zh' ? path.descriptionZh : path.descriptionEn;
            const pct = path.total > 0 ? Math.round((path.solved / path.total) * 100) : 0;
            const tag = `PATH_${String(idx + 1).padStart(2, '0')}`;

            return (
              <Link
                key={path.id}
                href={`/paths/${path.id}`}
                className="flex flex-col gap-[18px] p-7 min-h-[300px] relative cursor-pointer transition-colors duration-[180ms] group"
                style={{ background: 'var(--bg-elev)' }}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'color-mix(in oklab, var(--accent) 3%, var(--bg-elev))')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--bg-elev)')}
              >
                <div className="flex items-baseline justify-between">
                  <span className="mono text-[11px] text-text-3 tracking-[0.14em]">{tag}</span>
                  <span className="mono text-xs text-text-2 tabular-nums">{path.solved}/{path.total} · {pct}%</span>
                </div>
                <h3 className="text-[22px] font-semibold tracking-[-0.02em] leading-[1.2]">{title}</h3>
                <p className="text-sm text-text-2 leading-relaxed max-w-[50ch]">{desc}</p>

                {/* Sequence nodes */}
                <div className="flex items-center gap-0 mt-auto overflow-hidden">
                  {Array.from({ length: path.total }).map((_, i) => (
                    <span key={i} className="flex items-center">
                      <span
                        className="w-[22px] h-[22px] rounded-[6px] inline-flex items-center justify-center mono text-[9.5px] flex-shrink-0"
                        style={
                          i < path.solved
                            ? { background: 'var(--easy)', color: '#fff', border: '1px solid transparent' }
                            : i === path.solved
                            ? { background: 'var(--accent-wash)', borderColor: 'var(--accent)', color: 'var(--accent)', fontWeight: 500, border: '1px solid var(--accent)' }
                            : { background: 'var(--bg-sunken)', border: '1px solid var(--line)', color: 'var(--text-3)' }
                        }
                      >
                        {String.fromCharCode(65 + (i % 26))}
                      </span>
                      {i < path.total - 1 && (
                        <span className="w-[10px] h-px flex-shrink-0" style={{ background: 'var(--line)' }} />
                      )}
                    </span>
                  ))}
                </div>

                {/* Progress row */}
                <div
                  className="flex items-center gap-3 pt-4 mono text-xs text-text-2"
                  style={{ borderTop: '1px dashed var(--line)' }}
                >
                  <span>{t('progress')}</span>
                  <div className="flex-1 h-[3px] rounded-pill relative" style={{ background: 'var(--line)' }}>
                    <div
                      className="absolute inset-0 rounded-pill"
                      style={{ width: `${pct}%`, background: 'var(--accent)' }}
                    />
                  </div>
                  <span className="tabular-nums">{pct}%</span>
                  <span className="text-text-3 transition-[color,transform] duration-150 group-hover:text-accent group-hover:translate-x-[3px]">
                    <ArrowRight className="w-4 h-4" />
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </main>
      <Footer />
    </div>
  );
}
