import { NextResponse } from 'next/server';
import problems from '@/lib/problems.json';
import type { Problem, ProblemSummary } from '@/lib/types';

export async function GET() {
  // List views (catalog, drawer) only need metadata: including descriptions and hints would
  // take the payload from ~40 KB to ~537 KB. The detail page fetches /api/problems/[id].
  const list: ProblemSummary[] = (problems.problems as Problem[]).map(({
    id, title, titleZh, difficulty, track, category, categoryZh, order, leetcodeId, leetcodeSlug,
  }) => ({
    id, title, titleZh, difficulty, track, category, categoryZh, order, leetcodeId, leetcodeSlug,
  }));
  // Static build output: safe to cache in the browser for an hour
  return NextResponse.json(
    { problems: list, total: list.length },
    { headers: { 'Cache-Control': 'public, max-age=3600' } },
  );
}
