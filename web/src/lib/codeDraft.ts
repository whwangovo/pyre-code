const DRAFT_KEY_PREFIX = 'pyre-code-draft:';

function getDraftKey(problemId: string) {
  return `${DRAFT_KEY_PREFIX}${problemId}`;
}

export function loadCodeDraft(problemId: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(getDraftKey(problemId));
  } catch {
    return null;
  }
}

export function saveCodeDraft(problemId: string, code: string) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(getDraftKey(problemId), code);
  } catch {
    // Ignore storage write failures (private mode, quota, etc).
  }
}
