export interface Test {
  name: string;
  code: string;
}

export interface Problem {
  id: string;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  functionName: string;
  hint: string;
  hintZh: string;
  descriptionEn: string;
  descriptionZh: string;
  tests: Test[];
}

export interface TestResult {
  name: string;
  passed: boolean;
  execTimeMs: number;
  error?: string;
  output?: string;
}

export interface SubmissionResult {
  passed: number;
  total: number;
  allPassed: boolean;
  results: TestResult[];
  totalTimeMs: number;
  error?: string;
}

export interface ProblemProgress {
  status: 'todo' | 'attempted' | 'solved';
  bestTimeMs?: number;
  attempts: number;
  solvedAt?: string;
}

export interface ProgressMap {
  [taskId: string]: ProblemProgress;
}

export interface CustomTest {
  name: string;
  code: string;
}

export interface LearningPath {
  id: string;
  titleEn: string;
  titleZh: string;
  descriptionEn: string;
  descriptionZh: string;
  icon: string;
  problems: string[];
  prerequisites: string[];
}
