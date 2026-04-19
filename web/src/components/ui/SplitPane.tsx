'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface SplitPaneProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultRatio?: number; // 0-1, default 0.5
  minLeft?: number;      // px
  minRight?: number;     // px
  className?: string;
}

export function SplitPane({
  left,
  right,
  defaultRatio = 0.5,
  minLeft = 320,
  minRight = 380,
  className,
}: SplitPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [ratio, setRatio] = useState(defaultRatio);
  const dragging = useRef(false);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const newRatio = Math.max(
        minLeft / rect.width,
        Math.min(1 - minRight / rect.width, x / rect.width)
      );
      setRatio(newRatio);
    };
    const onMouseUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [minLeft, minRight]);

  return (
    <div ref={containerRef} className={cn('flex h-full w-full', className)}>
      <div style={{ width: `${ratio * 100}%` }} className="overflow-auto">
        {left}
      </div>
      <div
        onMouseDown={onMouseDown}
        className="w-1 cursor-col-resize hover:bg-accent/30 transition-colors flex-shrink-0"
        style={{ background: 'var(--line)' }}
      />
      <div style={{ width: `${(1 - ratio) * 100}%` }} className="overflow-auto">
        {right}
      </div>
    </div>
  );
}

interface VerticalSplitPaneProps {
  top: React.ReactNode;
  bottom: React.ReactNode;
  defaultRatio?: number; // 0-1, default 0.6 (60% top)
  minTop?: number;       // px
  minBottom?: number;    // px
  className?: string;
}

export function VerticalSplitPane({
  top,
  bottom,
  defaultRatio = 0.6,
  minTop = 200,
  minBottom = 150,
  className,
}: VerticalSplitPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [ratio, setRatio] = useState(defaultRatio);
  const dragging = useRef(false);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const y = e.clientY - rect.top;
      const newRatio = Math.max(
        minTop / rect.height,
        Math.min(1 - minBottom / rect.height, y / rect.height)
      );
      setRatio(newRatio);
    };
    const onMouseUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [minTop, minBottom]);

  return (
    <div ref={containerRef} className={cn('flex flex-col h-full w-full', className)}>
      <div style={{ height: `${ratio * 100}%` }} className="overflow-hidden min-h-0">
        {top}
      </div>
      <div
        onMouseDown={onMouseDown}
        className="h-1 cursor-row-resize hover:bg-accent/30 transition-colors flex-shrink-0"
        style={{ background: 'var(--line)' }}
      />
      <div style={{ height: `${(1 - ratio) * 100}%` }} className="overflow-hidden min-h-0">
        {bottom}
      </div>
    </div>
  );
}
