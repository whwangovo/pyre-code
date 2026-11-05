import type { Locale } from '@/lib/i18n';

export interface Test {
  name: string;
  code: string;
}

export interface Problem {
  id: string;
  title: string;
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  functionName: string;
  hint: string;
  hintZh: string;
  descriptionEn: string;
  descriptionZh: string;
  tests: Test[];
  /** Absent for the AI track */
  track?: 'hot100';
  category?: string;
  categoryZh?: string;
  /** Position in the Hot 100 study order */
  order?: number;
  leetcodeId?: number;
  leetcodeSlug?: string;
  acm?: { inputEn: string; inputZh: string; outputEn: string; outputZh: string };
  cases?: Record<CodeMode, AlgoCase[]>;
  starters?: Record<CodeMode, string>;
}

/** Hot 100 answer format: LeetCode-style class method, or a full stdin/stdout program */
export type CodeMode = 'core' | 'acm';

/** The list payload from /api/problems: metadata only, no descriptions, hints or tests. */
export interface ProblemSummary {
  id: string;
  title: string;
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  track?: 'hot100';
  category?: string;
  categoryZh?: string;
  /** Position in the Hot 100 study order */
  order?: number;
  leetcodeId?: number;
  leetcodeSlug?: string;
}

export interface AlgoCase {
  name: string;
  /** Large generated inputs are graded but not shown */
  hidden?: boolean;
  input?: string;
  expected?: string;
}

export interface TestResult {
  name: string;
  passed: boolean;
  execTimeMs: number;
  error?: string;
  output?: string;
  input?: string | null;
  expected?: string | null;
  actual?: string | null;
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
  titleZh: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  status: 'todo' | 'attempted' | 'solved';
}

export interface SubmissionHistory {
  id: number;
  passed: boolean;
  execTimeMs: number | null;
  submittedAt: string;
  code: string;
  mode?: CodeMode;
  note?: string;
}

export interface AiHelpConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  includeUserCode: boolean;
}

export interface AiChatMessage {
  role: 'user' | 'assistant';
  content: string;
  /** Short text shown in the bubble instead of content (used for quick-action prompts). */
  label?: string;
}

export interface AiHelpRequest {
  problemId: string;
  problemTitle: string;
  functionName: string;
  description: string;
  sampleTests: Array<{ name: string; code: string }>;
  messages: Array<Pick<AiChatMessage, 'role' | 'content'>>;
  userCode?: string;
  /** Hot 100 problems get an algorithm-interview persona instead of the LLM tutor */
  track?: 'hot100';
  mode?: CodeMode;
  locale: Locale;
  config: Omit<AiHelpConfig, 'includeUserCode'>;
}
