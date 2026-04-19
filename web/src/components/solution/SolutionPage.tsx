'use client';

import Link from 'next/link';
import { CodeEditor } from '@/components/workspace/CodeEditor';
import { useLocale } from '@/context/LocaleContext';
import solutionsData from '@/lib/solutions.json';
import { getSolutionCode } from '@/lib/problemContext';

interface Cell {
  type: string;
  source: string;
  role: string;
}

interface SolutionPageProps {
  problemId: string;
}

export function SolutionPageContent({ problemId }: SolutionPageProps) {
  const { t } = useLocale();
  const data = (solutionsData as Record<string, { cells: Cell[] }>)[problemId];
  const cells = data?.cells ?? [];

  if (cells.length === 0) {
    return <p className="text-sm text-text-3 p-6">{t('noSolution')}</p>;
  }

  const solutionCode = getSolutionCode(problemId);
  const explanations = cells.filter((c) => c.role === 'explanation');
  const demoCode = cells.filter((c) => c.role === 'demo').map((c) => c.source).join('\n\n');

  return (
    <div>
      {/* Breadcrumbs */}
      <div className="mono text-xs text-text-3 flex items-center gap-1.5 mb-5">
        <Link href="/problems" className="hover:text-text transition-colors">{t('problems')}</Link>
        <span className="opacity-60">/</span>
        <Link href={`/problems/${problemId}`} className="hover:text-text transition-colors">{problemId}</Link>
        <span className="opacity-60">/</span>
        <span className="text-text font-medium">{t('solution')}</span>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_260px] max-[900px]:grid-cols-1 gap-10 items-start">
        <div>
          <h1 className="text-[clamp(28px,3.5vw,44px)] font-semibold tracking-[-0.028em] leading-[1.08] mb-4">
            {t('solution')}
          </h1>

          {/* Solution code */}
          {solutionCode && (
            <div className="rounded-xl overflow-hidden mb-[18px]" style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}>
              <div
                className="flex items-center gap-2.5 px-3.5 h-[38px] mono text-xs text-text-2"
                style={{ borderBottom: '1px solid var(--line)', background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))' }}
              >
                <span className="text-text">{problemId}.py</span>
                <span className="flex-1" />
              </div>
              <div className="h-[400px]">
                <CodeEditor value={solutionCode} onChange={() => {}} readOnly />
              </div>
            </div>
          )}

          {/* Explanations */}
          {explanations.map((c, i) => (
            <p key={i} className="text-[14.5px] text-text-2 leading-[1.7] mb-3.5 max-w-[68ch]">{c.source}</p>
          ))}

          {/* Demo code */}
          {demoCode && (
            <div className="rounded-xl overflow-hidden mt-7" style={{ border: '1px solid var(--line)', background: 'var(--bg-elev)' }}>
              <div
                className="flex items-center gap-2.5 px-3.5 h-[38px] mono text-xs text-text-2"
                style={{ borderBottom: '1px solid var(--line)', background: 'color-mix(in oklab, var(--text) 2%, var(--bg-elev))' }}
              >
                <span className="text-text">demo</span>
              </div>
              <div style={{ height: `${Math.max(120, demoCode.split('\n').length * 22 + 32)}px` }}>
                <CodeEditor value={demoCode} onChange={() => {}} readOnly />
              </div>
            </div>
          )}
        </div>

        {/* TOC sidebar */}
        <aside className="sticky top-[76px] hidden md:block">
          <h5 className="eyebrow mb-3">{t('solution')}</h5>
          <div className="flex flex-col gap-1">
            <Link
              href={`/problems/${problemId}`}
              className="text-[13px] text-text-2 hover:text-text transition-colors px-2.5 py-1.5 rounded-[7px] hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)]"
            >
              ← {t('backToProblem')}
            </Link>
          </div>
        </aside>
      </div>
    </div>
  );
}
