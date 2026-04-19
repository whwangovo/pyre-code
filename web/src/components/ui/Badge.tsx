import { cn } from '@/lib/utils';

const difficultyStyles: Record<string, string> = {
  easy: 'text-easy',
  medium: 'text-medium',
  hard: 'text-hard',
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'easy' | 'medium' | 'hard' | 'default';
  className?: string;
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 mono text-[11px] px-[7px] py-[2px] rounded-[5px] tracking-[0.04em]',
        difficultyStyles[variant] ?? 'text-text-2',
        className,
      )}
      style={
        variant !== 'default'
          ? {
              border: `1px solid color-mix(in oklab, var(--${variant}) 30%, var(--line))`,
              background: `color-mix(in oklab, var(--${variant}) 7%, var(--bg-elev))`,
            }
          : {
              border: '1px solid var(--line)',
              background: 'var(--bg-sunken)',
            }
      }
    >
      {children}
    </span>
  );
}
