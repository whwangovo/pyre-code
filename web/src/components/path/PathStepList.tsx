'use client';

import Link from 'next/link';
import { CheckCircle2, Circle, ChevronRight } from 'lucide-react';
import { useLocale } from '@/context/LocaleContext';
import { DifficultyBadge } from '@/components/problem/DifficultyBadge';

interface PathStep {
  id: string;
  title: string;
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  status: 'todo' | 'attempted' | 'solved';
}

interface PathStepListProps {
  pathId: string;
  steps: PathStep[];
}

export function PathStepList({ pathId, steps }: PathStepListProps) {
  const { locale } = useLocale();

  return (
    <ol className="relative">
      {steps.map((step, i) => {
        const solved = step.status === 'solved';
        const title = locale === 'zh' ? step.titleZh : step.title;
        return (
          <li key={step.id} className="flex gap-4 pb-0">
            {/* connector line */}
            <div className="flex flex-col items-center">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 z-10 ${solved ? 'text-green-500' : 'text-gray-300'}`}>
                {solved
                  ? <CheckCircle2 className="w-7 h-7" />
                  : <Circle className="w-7 h-7" />
                }
              </div>
              {i < steps.length - 1 && (
                <div className="w-px flex-1 bg-border my-1" />
              )}
            </div>
            {/* content */}
            <div className={`flex-1 pb-6 ${i === steps.length - 1 ? 'pb-0' : ''}`}>
              <Link
                href={`/problems/${step.id}?path=${pathId}`}
                className="group flex items-center justify-between rounded-xl border border-border bg-white px-4 py-3 hover:border-accent/40 hover:shadow-sm transition-all duration-200"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xs text-text-tertiary w-5 text-right">{i + 1}</span>
                  <span className="text-sm font-medium text-text-primary">{title}</span>
                  <DifficultyBadge difficulty={step.difficulty} />
                </div>
                <ChevronRight className="w-4 h-4 text-text-tertiary group-hover:text-accent transition-colors" />
              </Link>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
