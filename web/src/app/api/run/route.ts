import { NextResponse } from 'next/server';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import type { SubmissionResult } from '@/lib/types';

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { taskId?: unknown; code?: unknown; testIndices?: unknown; mode?: unknown } | null;
  const { taskId, code, testIndices, mode } = body ?? {};
  if (typeof taskId !== 'string' || typeof code !== 'string') {
    return NextResponse.json({ error: 'Missing taskId or code' }, { status: 400 });
  }

  try {
    const res = await fetch(`${GRADING_SERVICE_URL}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskId, code, testIndices, mode }),
    });
    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: errText },
        { status: 502 }
      );
    }
    const result: SubmissionResult = await res.json();
    return NextResponse.json(result);
  } catch {
    return NextResponse.json(
      { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: 'Grading service unreachable' },
      { status: 502 }
    );
  }
}
