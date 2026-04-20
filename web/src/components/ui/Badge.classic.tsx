import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded-pill px-2.5 py-0.5 text-xs font-medium',
  {
    variants: {
      variant: {
        easy: 'bg-easy/10 text-easy',
        medium: 'bg-medium/10 text-medium',
        hard: 'bg-hard/10 text-hard',
        solved: 'bg-solved/10 text-solved',
        default: 'bg-gray-100 text-text-secondary',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

interface BadgeProps extends VariantProps<typeof badgeVariants> {
  children: React.ReactNode;
  className?: string;
}

export function Badge({ children, variant, className }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)}>
      {children}
    </span>
  );
}
