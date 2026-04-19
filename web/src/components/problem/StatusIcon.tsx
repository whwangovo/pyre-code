import type { ProblemProgress } from '@/lib/types';

interface StatusIconProps {
  status: ProblemProgress['status'];
}

export function StatusIcon({ status }: StatusIconProps) {
  if (status === 'solved') {
    return (
      <span className="w-5 h-5 inline-flex items-center justify-center rounded-full text-easy">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </span>
    );
  }
  if (status === 'attempted') {
    return (
      <span className="w-5 h-5 inline-flex items-center justify-center rounded-full text-medium">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </span>
    );
  }
  return <span className="w-5 h-5 inline-flex items-center justify-center" />;
}
