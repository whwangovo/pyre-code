'use client';

import { CodeEditor } from './CodeEditor';
import { useLocale } from '@/context/LocaleContext';
import solutionsData from '@/lib/solutions.json';
import { getSolutionCode } from '@/lib/problemContext';

interface Cell {
  type: string;
  source: string;
  role: string;
}

interface SolutionTabProps {
  problemId: string;
}

export function SolutionTab({ problemId }: SolutionTabProps) {
  const { t } = useLocale();
  const data = (solutionsData as Record<string, { cells: Cell[] }>)[problemId];
  const cells = data?.cells ?? [];

  if (cells.length === 0) {
    return <div className="p-6 text-sm text-text-tertiary">{t('noSolution')}</div>;
  }

  const solutionCode = getSolutionCode(problemId);

  const demoCode = cells
    .filter((c) => c.role === 'demo')
    .map((c) => c.source)
    .join('\n\n');

  const explanations = cells.filter((c) => c.role === 'explanation');

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {solutionCode && (
        <div className="flex flex-col gap-1">
          <p className="text-xs font-medium text-text-secondary px-1">Solution</p>
          <div className="rounded-lg overflow-hidden border border-border/50">
            <CodeEditor
              value={solutionCode}
              onChange={() => {}}
              readOnly
              height={`${Math.max(120, solutionCode.split('\n').length * 22 + 32)}px`}
              allowParentScrollOnWheel
            />
          </div>
        </div>
      )}
      {explanations.map((c, i) => (
        <p key={i} className="text-sm text-text-secondary px-1 whitespace-pre-wrap">{c.source}</p>
      ))}
      {demoCode && (
        <div className="flex flex-col gap-1">
          <p className="text-xs font-medium text-text-secondary px-1">Demo</p>
          <div className="rounded-lg overflow-hidden border border-border/50">
            <CodeEditor
              value={demoCode}
              onChange={() => {}}
              readOnly
              height={`${Math.max(80, demoCode.split('\n').length * 22 + 32)}px`}
              allowParentScrollOnWheel
            />
          </div>
        </div>
      )}
    </div>
  );
}
