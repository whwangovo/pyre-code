import { NextResponse } from 'next/server';
import pathsData from '@/lib/paths.json';
import problemsData from '@/lib/problems.json';
import { gradingFetch } from '@/lib/grading';
import { getSessionToken } from '@/lib/session';
import type { LearningPath } from '@/lib/types';

const difficultyOf = new Map(
  (problemsData.problems as Array<{ id: string; difficulty: string }>).map((p) => [p.id, p.difficulty]),
);

export async function GET() {
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

  const paths = (pathsData.paths as LearningPath[]).map((path) => {
    const solved = path.problems.filter((id) => progressMap[id]?.status === 'solved').length;
    const count = (difficulty: string) => path.problems.filter((id) => difficultyOf.get(id) === difficulty).length;
    return {
      ...path,
      solved,
      total: path.problems.length,
      easy: count('Easy'),
      medium: count('Medium'),
      hard: count('Hard'),
    };
  });

  return NextResponse.json({ paths });
}
