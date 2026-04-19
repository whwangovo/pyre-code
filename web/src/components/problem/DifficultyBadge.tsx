import { Badge } from '@/components/ui/Badge';
import type { Problem } from '@/lib/types';

interface DifficultyBadgeProps {
  difficulty: Problem['difficulty'];
}

export function DifficultyBadge({ difficulty }: DifficultyBadgeProps) {
  const variant = difficulty.toLowerCase() as 'easy' | 'medium' | 'hard';
  return <Badge variant={variant}>{difficulty.toUpperCase()}</Badge>;
}
