'use client';

import { useRef } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import { appleLight } from '@/lib/monacoTheme';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  height?: string;
  allowParentScrollOnWheel?: boolean;
}

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  height = '100%',
  allowParentScrollOnWheel = false,
}: CodeEditorProps) {
  const editorRef = useRef<any>(null);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monaco.editor.defineTheme('apple-light', appleLight);
    monaco.editor.setTheme('apple-light');
  };

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
        fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
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
