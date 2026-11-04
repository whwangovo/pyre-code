import { NextResponse } from 'next/server';
import pathsData from '@/lib/paths.json';
import problemsData from '@/lib/problems.json';
import { gradingFetch } from '@/lib/grading';
import { getSessionToken } from '@/lib/session';
import type { LearningPath } from '@/lib/types';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const path = (pathsData.paths as LearningPath[]).find((p) => p.id === id);
  if (!path) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const sessionToken = await getSessionToken();

  let progressMap: Record<string, { status: string }> = {};
  if (sessionToken) {
    try {
      const res = await gradingFetch(sessionToken, '/progress');
      if (res.ok) progressMap = await res.json();
    } catch {
      // continue without progress
    }
  }

  const problems = path.problems.map((problemId) => {
    const problem = (problemsData as { problems: Array<{ id: string; title: string; titleZh: string; difficulty: string }> }).problems.find((p) => p.id === problemId);
    return {
      id: problemId,
      title: problem?.title ?? problemId,
      titleZh: problem?.titleZh ?? problemId,
      difficulty: (problem?.difficulty ?? 'Easy') as 'Easy' | 'Medium' | 'Hard',
      status: (progressMap[problemId]?.status ?? 'todo') as 'todo' | 'attempted' | 'solved',
    };
  });

  const solved = problems.filter((p) => p.status === 'solved').length;

  return NextResponse.json({ ...path, problems, solved, total: path.problems.length });
}
