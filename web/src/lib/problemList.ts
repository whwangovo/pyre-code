import type { ProblemSummary } from '@/lib/types';

// The list is static build output (only changes on deploy), but the workspace refetches it on
// every problem switch and the catalog on every visit. One module-level promise per page load
// keeps later calls free.
let cache: Promise<ProblemSummary[]> | null = null;

export function loadProblemList(): Promise<ProblemSummary[]> {
  cache ??= fetch('/api/problems')
    .then((r) => r.json())
    .then((d: { problems: ProblemSummary[] }) => d.problems)
    .catch((err) => {
      // Failed fetch: clear the cache so a later call retries
      cache = null;
      throw err;
    });
  return cache;
}
