'use client';

import { CodeEditor } from '@/components/workspace/CodeEditor.classic';
import solutionsData from '@/lib/solutions.json';

interface SolutionPageProps {
  problemId: string;
}

interface Cell {
  type: 'code' | 'markdown';
  source: string;
}

export function SolutionPageContent({ problemId }: SolutionPageProps) {
  const data = (solutionsData as Record<string, { cells: Cell[] }>)[problemId];
  const cells = data?.cells ?? [];

  if (cells.length === 0) {
    return <p className="text-sm text-text-tertiary p-6">No solution available yet.</p>;
  }

  const codeCells = cells.filter((c) => c.type === 'code');
  const markdownCells = cells.filter((c) => c.type === 'markdown');
  const code = codeCells.map((c) => c.source).join('\n\n');

  return (
    <div className="grid grid-cols-2 gap-6 h-[calc(100vh-8rem)]">
      <div className="rounded-xl border border-border overflow-hidden">
        <CodeEditor value={code} onChange={() => {}} readOnly />
      </div>
      <div className="overflow-auto space-y-4">
        {markdownCells.map((cell, i) => (
          <div key={i} className="p-4 rounded-xl bg-surface-secondary border border-border/50">
            <pre className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed">
              {cell.source}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
