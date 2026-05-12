'use client';

import { useRef, useEffect } from 'react';
import dynamic from 'next/dynamic';
import type { OnMount } from '@monaco-editor/react';
import { appleLight, appleDark } from '@/lib/monacoTheme';
import { useTheme } from '@/context/ThemeContext';

const Editor = dynamic(() => import('@monaco-editor/react').then((m) => m.default), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-bg-sunken animate-pulse" />,
});

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
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<any>(null);
  const runRef = useRef(onRunShortcut);
  const submitRef = useRef(onSubmitShortcut);
  const { theme } = useTheme();

  useEffect(() => {
    runRef.current = onRunShortcut;
    submitRef.current = onSubmitShortcut;
  });

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    monaco.editor.defineTheme('apple-light', appleLight);
    monaco.editor.defineTheme('apple-dark', appleDark);
    monaco.editor.setTheme(theme === 'dark' ? 'apple-dark' : 'apple-light');
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
        readOnly,
        fontSize: 14,
        lineHeight: 22,
        fontFamily: "'Geist Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        padding: { top: 16, bottom: 16 },
        renderLineHighlight: 'line',
        smoothScrolling: true,
        cursorBlinking: 'smooth',
        cursorSmoothCaretAnimation: 'on',
        bracketPairColorization: { enabled: true },
        overviewRulerBorder: false,
        hideCursorInOverviewRuler: true,
        scrollbar: {
          alwaysConsumeMouseWheel: !allowParentScrollOnWheel,
          verticalScrollbarSize: 6,
          horizontalScrollbarSize: 6,
        },
      }}
    />
  );
}
