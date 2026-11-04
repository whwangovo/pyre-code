import { GRADING_SERVICE_URL } from '@/lib/constants';

/**
 * Fetches from the grading service as the learner behind `sessionToken`. The service
 * resolves the user from the token itself; reads used to pass an enumerable numeric id.
 */
export function gradingFetch(
  sessionToken: string,
  path: string,
  init?: Omit<RequestInit, 'headers'> & { headers?: Record<string, string> },
): Promise<Response> {
  return fetch(`${GRADING_SERVICE_URL}${path}`, {
    ...init,
    headers: { ...init?.headers, 'X-Session-Token': sessionToken },
  });
}
