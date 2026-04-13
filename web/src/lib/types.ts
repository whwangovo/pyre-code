import type { Locale } from '@/lib/i18n';

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

export interface LearningPathProblemSummary {
  id: string;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  status: 'todo' | 'attempted' | 'solved';
}

export interface SubmissionHistory {
  id: number;
  passed: boolean;
  execTimeMs: number | null;
  submittedAt: string;
  code: string;
}

export interface AiHelpConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  includeUserCode: boolean;
}

export interface AiHelpRequest {
  problemId: string;
  problemTitle: string;
  functionName: string;
  description: string;
  solutionCode: string;
  sampleTests: Array<{ name: string; code: string }>;
  customPrompt?: string;
  userCode?: string;
  locale: Locale;
  config: Omit<AiHelpConfig, 'includeUserCode'>;
}

export interface AiHelpResponse {
  guidance: string;
  model?: string;
}
