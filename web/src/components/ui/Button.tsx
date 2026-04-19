import { cn } from '@/lib/utils';

const variants = {
  primary: 'bg-[var(--text)] text-[var(--bg)] border-[var(--text)] hover:bg-[color-mix(in_oklab,var(--text)_88%,var(--accent))]',
  secondary: 'bg-bg-elev text-text border-line hover:border-line-strong',
  ghost: 'bg-transparent text-text-2 border-transparent hover:text-text hover:bg-[color-mix(in_oklab,var(--text)_5%,transparent)]',
};

const sizes = {
  sm: 'h-8 px-3 text-xs',
  md: 'h-9 px-3.5 text-[13.5px]',
  lg: 'h-11 px-6 text-base',
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variants;
  size?: keyof typeof sizes;
}

export function Button({ className, variant = 'primary', size = 'md', ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 font-medium rounded-[9px] cursor-pointer border transition-[transform,background,border-color,color] duration-150 disabled:opacity-50 disabled:pointer-events-none',
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}
