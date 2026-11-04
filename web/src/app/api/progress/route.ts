import { NextResponse } from 'next/server';
import { getSessionToken } from '@/lib/session';
import { gradingFetch } from '@/lib/grading';

export async function GET() {
  const sessionToken = await getSessionToken();
  if (!sessionToken) return NextResponse.json({ progress: {} });

  try {
    // The grading service resolves the user from the token; unknown tokens read as empty
    const res = await gradingFetch(sessionToken, '/progress');
    if (!res.ok) return NextResponse.json({ progress: {} });
    return NextResponse.json({ progress: await res.json() });
  } catch {
    return NextResponse.json({ progress: {} });
  }
}
