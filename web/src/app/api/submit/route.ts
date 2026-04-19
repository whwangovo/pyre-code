import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { GRADING_SERVICE_URL } from '@/lib/constants';
import { SubmissionResult } from '@/lib/types';

export async function POST(request: Request) {
  const { taskId, code } = await request.json();

  const cookieStore = await cookies();
  let sessionToken = cookieStore.get('session_token')?.value;
  if (!sessionToken) {
    sessionToken = crypto.randomUUID();
  }

  // Call grading service
  const gradingResponse = await fetch(`${GRADING_SERVICE_URL}/grade`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskId, code }),
  });

  if (!gradingResponse.ok) {
    const errText = await gradingResponse.text();
    return NextResponse.json(
      { passed: 0, total: 0, allPassed: false, results: [], totalTimeMs: 0, error: errText },
      { status: 502 }
    );
  }

  const result: SubmissionResult = await gradingResponse.json();

  // Ensure the anonymous user exists before saving progress/submissions.
  const userResponse = await fetch(`${GRADING_SERVICE_URL}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken }),
  });

  if (!userResponse.ok) {
    const errText = await userResponse.text();
    return NextResponse.json(
      {
        ...result,
        error: `Failed to initialize user session: ${errText}`,
      },
      { status: 502 }
    );
  }

  // Save progress
  const status = result.allPassed ? 'solved' : 'attempted';
  const progressResponse = await fetch(`${GRADING_SERVICE_URL}/progress`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionToken, taskId, status, execTimeMs: result.totalTimeMs, code, allPassed: result.allPassed }),
  });

  if (!progressResponse.ok) {
    const errText = await progressResponse.text();
    return NextResponse.json(
      {
        ...result,
        error: `Failed to save submission history: ${errText}`,
      },
      { status: 502 }
    );
  }

  const response = NextResponse.json(result);
  response.cookies.set('session_token', sessionToken, {
    httpOnly: true,
    maxAge: 60 * 60 * 24 * 30,
  });

  return response;
}
