'use client';

import { useRef, useEffect } from 'react';
import dynamic from 'next/dynamic';
import type { Monaco, OnMount } from '@monaco-editor/react';
import { appleLight, appleDark } from '@/lib/monacoTheme';
import { useTheme } from '@/context/ThemeContext';

const Editor = dynamic(() => import('@monaco-editor/react').then((m) => m.default), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-bg-sunken animate-pulse" />,
});

const DiffEditor = dynamic(() => import('@monaco-editor/react').then((m) => m.DiffEditor), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-bg-sunken animate-pulse" />,
});

const sharedOptions = {
  fontSize: 14,
  lineHeight: 22,
  fontFamily: "'Geist Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  overviewRulerBorder: false,
  hideCursorInOverviewRuler: true,
};

function applyTheme(monaco: Monaco, theme: string) {
  monaco.editor.defineTheme('apple-light', appleLight);
  monaco.editor.defineTheme('apple-dark', appleDark);
  monaco.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
}

/** Read-only diff: `original` on the left, `modified` on the right (inline when narrow). */
export function CodeDiffViewer({ original, modified }: { original: string; modified: string }) {
  const monacoRef = useRef<Monaco | null>(null);
  const { theme } = useTheme();

  useEffect(() => {
    if (monacoRef.current) applyTheme(monacoRef.current, theme);
  }, [theme]);

  return (
    <DiffEditor
      height="100%"
      language="python"
      original={original}
      modified={modified}
      onMount={(_editor, monaco) => {
        monacoRef.current = monaco;
        applyTheme(monaco, theme);
      }}
      options={{
        ...sharedOptions,
        readOnly: true,
        originalEditable: false,
        renderSideBySide: true,
        useInlineViewWhenSpaceIsLimited: true,
        renderOverviewRuler: false,
        scrollbar: { verticalScrollbarSize: 6, horizontalScrollbarSize: 6 },
      }}
    />
  );
}

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  height?: string;
  allowParentScrollOnWheel?: boolean;
  onRunShortcut?: () => void;
  onSubmitShortcut?: () => void;
}

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  height = '100%',
  allowParentScrollOnWheel = false,
  onRunShortcut,
  onSubmitShortcut,
}: CodeEditorProps) {
  const monacoRef = useRef<Monaco | null>(null);
  const runRef = useRef(onRunShortcut);
  const submitRef = useRef(onSubmitShortcut);
  const { theme } = useTheme();

  useEffect(() => {
    runRef.current = onRunShortcut;
    submitRef.current = onSubmitShortcut;
  });

  const handleMount: OnMount = (editor, monaco) => {
    monacoRef.current = monaco;
    applyTheme(monaco, theme);
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => runRef.current?.());
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => submitRef.current?.());
  };

  useEffect(() => {
    if (monacoRef.current) {
      monacoRef.current.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
    }
  }, [theme]);

  return (
    <Editor
      height={height}
      language="python"
      value={value}
      onChange={(v) => onChange(v || '')}
      onMount={handleMount}
      options={{
        ...sharedOptions,
        readOnly,
        padding: { top: 16, bottom: 16 },
        renderLineHighlight: 'line',
        smoothScrolling: true,
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        bracketPairColorization: { enabled: true },
        scrollbar: {
          alwaysConsumeMouseWheel: !allowParentScrollOnWheel,
          verticalScrollbarSize: 6,
          horizontalScrollbarSize: 6,
        },
      }}
    />
  );
}
