'use client';

import { useEffect, useState, useMemo } from 'react';
import Link from 'next/link';
import { Search } from 'lucide-react';
import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { StatusIcon } from '@/components/problem/StatusIcon';
import { Badge } from '@/components/ui/Badge';
import { useLocale } from '@/context/LocaleContext';
import { useDesign } from '@/context/DesignContext';
import { ProblemsPageClassic } from '@/components/problems/ProblemsPage.classic';
import { cn } from '@/lib/utils';
import type { Problem, ProgressMap } from '@/lib/types';

export default function ProblemsPage() {
  const { design } = useDesign();
  if (design === 'classic') return <ProblemsPageClassic />;
  return <ProblemsPageNew />;
}

function ProblemsPageNew() {
  const { t, tProblem } = useLocale();
  const [problems, setProblems] = useState<Problem[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [search, setSearch] = useState('');
  const [difficulty, setDifficulty] = useState('');
  const [category, setCategory] = useState('');

  useEffect(() => {
    fetch('/api/problems')
      .then((r) => r.json())
      .then((d) => setProblems(d.problems));
    fetch('/api/progress')
      .then((r) => r.json())
      .then((d) => setProgress(d.progress || {}));
  }, []);

  const solvedCount = Object.values(progress).filter((p) => p.status === 'solved').length;

  const categories = useMemo(() => {
    const map = new Map<string, number>();
    problems.forEach((p) => {
      const cat = (p as Problem & { category?: string }).category || 'Other';
      map.set(cat, (map.get(cat) || 0) + 1);
    });
    return Array.from(map.entries()).sort((a, b) => b[1] - a[1]);
  }, [problems]);

  const diffCounts = useMemo(() => ({
    Easy: problems.filter((p) => p.difficulty === 'Easy').length,
    Medium: problems.filter((p) => p.difficulty === 'Medium').length,
    Hard: problems.filter((p) => p.difficulty === 'Hard').length,
  }), [problems]);

  const filtered = problems.filter((p) => {
    if (search) {
      const s = search.toLowerCase();
      if (!p.title.toLowerCase().includes(s) && !p.id.toLowerCase().includes(s)) return false;
    }
    if (difficulty && p.difficulty !== difficulty) return false;
    if (category) {
      const cat = (p as Problem & { category?: string }).category || 'Other';
      if (cat !== category) return false;
    }
    return true;
  });

  return (
    <div className="min-h-screen bg-bg">
      <TopNav solvedCount={solvedCount} totalCount={problems.length} />
      <main className="max-w-[1280px] mx-auto px-7 pt-8 pb-20">
        {/* Page header */}
        <div className="flex items-end justify-between gap-8 mb-9 pb-6" style={{ borderBottom: '1px solid var(--line)' }}>
          <div>
            <div className="eyebrow mb-2">{t('problems')}</div>
            <h1 className="text-[clamp(28px,3vw,40px)] font-semibold tracking-[-0.03em] mb-2">{t('problems')}</h1>
            <p className="text-sm text-text-2 max-w-[52ch]">{t('problemsSubtitle')}</p>
          </div>
          <div className="flex gap-[18px]">
            {(['Easy', 'Medium', 'Hard'] as const).map((d) => (
              <div key={d} className="text-right">
                <div className="eyebrow">{t(d)}</div>
                <div className="text-xl tabular-nums mt-0.5" style={{ color: `var(--${d.toLowerCase()})` }}>
                  {diffCounts[d]}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-[232px_minmax(0,1fr)] gap-10 items-start max-[900px]:grid-cols-1">
          {/* Sidebar */}
          <aside className="sticky top-[76px]">
            {/* Categories */}
            {categories.length > 0 && (
              <div className="mb-6">
                <h5 className="eyebrow mb-2.5">{t('category')}</h5>
                <div className="flex flex-col gap-px">
                  <button
                    onClick={() => setCategory('')}
                    className={cn(
                      'flex items-center justify-between gap-2.5 px-2.5 py-[7px] rounded-[7px] text-[13.5px] cursor-pointer transition-[background,color] duration-150',
                      !category
                        ? 'text-accent'
                        : 'text-text-2 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)]'
                    )}
                    style={!category ? { background: 'var(--accent-wash)' } : undefined}
                  >
                    <span>{t('All')}</span>
                    <span className="mono text-[11.5px] text-text-3 tabular-nums">{problems.length}</span>
                  </button>
                  {categories.map(([cat, count]) => (
                    <button
                      key={cat}
                      onClick={() => setCategory(cat === category ? '' : cat)}
                      className={cn(
                        'flex items-center justify-between gap-2.5 px-2.5 py-[7px] rounded-[7px] text-[13.5px] cursor-pointer transition-[background,color] duration-150',
                        category === cat
                          ? 'text-accent'
                          : 'text-text-2 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)]'
                      )}
                      style={category === cat ? { background: 'var(--accent-wash)' } : undefined}
                    >
                      <span>{cat}</span>
                      <span className={cn('mono text-[11.5px] tabular-nums', category === cat ? 'text-accent opacity-80' : 'text-text-3')}>{count}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Difficulty filter */}
            <div>
              <h5 className="eyebrow mb-2.5">{t('difficulty')}</h5>
              <div className="flex flex-col gap-0.5">
                {(['Easy', 'Medium', 'Hard'] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setDifficulty(d === difficulty ? '' : d)}
                    className={cn(
                      'flex items-center gap-2.5 px-2.5 py-1.5 rounded-[7px] text-[13.5px] cursor-pointer transition-[background,color] duration-150',
                      difficulty === d
                        ? 'text-text bg-[color-mix(in_oklab,var(--text)_6%,transparent)]'
                        : 'text-text-2 hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)]'
                    )}
                  >
                    <span className="w-[7px] h-[7px] rounded-full" style={{ background: `var(--${d.toLowerCase()})` }} />
                    <span>{t(d)}</span>
                    <span className="ml-auto mono text-[11.5px] text-text-3 tabular-nums">{diffCounts[d]}</span>
                  </button>
                ))}
              </div>
            </div>
          </aside>

          {/* Main content */}
          <div>
            {/* Toolbar */}
            <div className="flex gap-2.5 mb-4 items-center">
              <div
                className="flex-1 flex items-center gap-2 px-3 h-[38px] rounded-[9px]"
                style={{ background: 'var(--bg-elev)', border: '1px solid var(--line)' }}
              >
                <Search className="w-3.5 h-3.5 text-text-3 flex-shrink-0" />
                <input
                  type="text"
                  placeholder={t('searchPlaceholder')}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="flex-1 bg-transparent text-sm text-text placeholder:text-text-3 outline-none"
                />
              </div>
            </div>

            {/* Table */}
            <div className="rounded-[12px] overflow-hidden" style={{ border: '1px solid var(--line)' }}>
              <table className="w-full text-sm">
                <thead>
                  <tr style={{ background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))' }}>
                    <th className="text-left px-4 py-2.5 mono text-[11px] text-text-3 tracking-[0.12em] uppercase font-medium w-12">#</th>
                    <th className="w-8" />
                    <th className="text-left px-2 py-2.5 mono text-[11px] text-text-3 tracking-[0.12em] uppercase font-medium">{t('problems')}</th>
                    <th className="text-left px-2 py-2.5 mono text-[11px] text-text-3 tracking-[0.12em] uppercase font-medium">{t('category')}</th>
                    <th className="text-left px-2 py-2.5 mono text-[11px] text-text-3 tracking-[0.12em] uppercase font-medium">{t('difficulty')}</th>
                    <th className="text-right px-2 py-2.5 mono text-[11px] text-text-3 tracking-[0.12em] uppercase font-medium">{t('time')}</th>
                    <th className="text-right px-4 py-2.5 mono text-[11px] text-text-3 tracking-[0.12em] uppercase font-medium">{t('status')}</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((p, i) => {
                    const ps = progress[p.id];
                    const status = ps?.status || 'todo';
                    const best = ps?.bestTimeMs;
                    const cat = (p as Problem & { category?: string }).category || '';
                    return (
                      <tr
                        key={p.id}
                        className="cursor-pointer transition-colors duration-150 hover:bg-[color-mix(in_oklab,var(--accent)_3%,var(--bg-elev))]"
                        style={{ borderTop: '1px solid var(--line)', background: 'var(--bg-elev)' }}
                        onClick={() => window.location.href = `/problems/${p.id}`}
                      >
                        <td className="px-4 py-3 mono text-[12px] text-text-3 tabular-nums">{String(i + 1).padStart(3, '0')}</td>
                        <td className="px-1 py-3"><StatusIcon status={status} /></td>
                        <td className="px-2 py-3">
                          <div className="font-medium text-text">{tProblem(p.id)}</div>
                          <div className="mono text-[11.5px] text-text-3 mt-0.5">{p.id}.py</div>
                        </td>
                        <td className="px-2 py-3 mono text-[12px] text-text-2">{cat}</td>
                        <td className="px-2 py-3">
                          <Badge variant={p.difficulty.toLowerCase() as 'easy' | 'medium' | 'hard'}>
                            {p.difficulty.toUpperCase()}
                          </Badge>
                        </td>
                        <td className="px-2 py-3 text-right mono text-[12px] text-text-3 tabular-nums">
                          {best ? `${best}ms` : '—'}
                        </td>
                        <td className="px-4 py-3 text-right text-[12.5px]">
                          <span className={cn(
                            status === 'solved' ? 'text-easy' : status === 'attempted' ? 'text-medium' : 'text-text-3'
                          )}>
                            {status === 'solved' ? t('Solved') : status === 'attempted' ? t('Attempted') : t('Todo')}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {filtered.length === 0 && (
                <p className="text-sm text-text-3 text-center py-12">{t('noMatch')}</p>
              )}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
