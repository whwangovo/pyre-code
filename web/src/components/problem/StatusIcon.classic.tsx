import { CheckCircle, Circle, Clock } from 'lucide-react';
import type { ProblemProgress } from '@/lib/types';

interface StatusIconProps {
  status: ProblemProgress['status'];
}

export function StatusIcon({ status }: StatusIconProps) {
  switch (status) {
    case 'solved':
      return <CheckCircle className="w-4 h-4 text-solved" />;
    case 'attempted':
      return <Clock className="w-4 h-4 text-medium" />;
    default:
      return <Circle className="w-4 h-4 text-gray-300" />;
  }
}
