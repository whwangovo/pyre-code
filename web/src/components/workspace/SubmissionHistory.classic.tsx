'use client';

import { useProblemStore } from '@/store/problemStore';
import { useLocale } from '@/context/LocaleContext';

export function SubmissionHistory() {
  const { submissionHistory, setCurrentCode } = useProblemStore();
  const { locale } = useLocale();

  if (!submissionHistory.length) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-text-tertiary">
        {locale === 'zh' ? '暂无提交记录' : 'No submissions yet'}
      </div>
    );
  }

  return (
    <div className="overflow-auto h-full">
      {submissionHistory.map((s) => (
        <div key={s.id} className="flex items-center gap-3 px-4 py-2 border-b border-border hover:bg-surface-secondary">
          <span className={`text-sm font-medium ${s.passed ? 'text-easy' : 'text-hard'}`}>
            {s.passed
              ? (locale === 'zh' ? '通过' : 'Accepted')
              : (locale === 'zh' ? '未通过' : 'Wrong Answer')}
          </span>
          {s.execTimeMs != null && (
            <span className="text-xs text-text-tertiary">{s.execTimeMs.toFixed(1)}ms</span>
          )}
          <span className="text-xs text-text-tertiary ml-auto">
            {new Date(s.submittedAt).toLocaleString()}
          </span>
          <button
            onClick={() => setCurrentCode(s.code)}
            className="text-xs text-accent hover:underline flex-shrink-0"
          >
            {locale === 'zh' ? '恢复' : 'Restore'}
          </button>
        </div>
      ))}
    </div>
  );
}
