1. task_record.json — the task's source data

  This is written first, directly from the input JSONL line. It's just the raw task record from gaia_lv1_x4.jsonl serialized to disk — question, expected answer, file
  attachments, metadata, etc. It's a snapshot of what the runner knows about the task before doing anything. Nothing is generated here; it's purely a copy of the
  input.

  ---
  2. task_prompt.txt — the formatted prompt

  The runner takes task_record.json and renders it into a human-readable prompt string. This involves injecting the question, the task ID, the file path (if there's
  an attachment), and the guidance rules into a template. task_prompt.txt is what will actually be passed to Claude as the user message. It's generated from the task
  record, not the other way around.

  ---
  3. cli_command.json — the exact subprocess call

  The runner constructs the claude -p ... shell command and saves it before launching. It records the full argument list: the rendered prompt, --output-format json,
  --json-schema (the structured output schema), --max-turns, --model, --allowedTools, --settings, --session-id, etc. This file exists for reproducibility and
  debugging — if a task fails, you can read cli_command.json and manually re-run the exact same command. It is generated from task_prompt.txt plus the runner's CLI
  flags, not the other way around.

  The order so far: task_record.json → task_prompt.txt → cli_command.json → subprocess is launched

  ---
  4. hook_events.jsonl — a live event log from inside Claude's session

  This is the most interesting one. Claude Code supports hooks: shell scripts that are called by the Claude process itself at specific lifecycle events during a
  session. The runner installs a hook script (tools/_claude_trace_hook.py) that Claude calls at each event. Each call appends one JSON line to hook_events.jsonl.

  The events follow Claude's internal lifecycle:
  - SessionStart — Claude process started, session ID assigned
  - UserPromptSubmit — the prompt was submitted to the model
  - PreToolUse — Claude is about to call a tool (e.g. WebSearch, Read, Bash), with the tool name and input captured
  - PostToolUse — tool returned, result captured
  - PostToolUseFailure — tool call failed (permission denied, exception, etc.)
  - Stop — Claude finished and produced a final response

  Because the hook script is a separate process invoked by Claude, it continues writing even if the runner's Python process is interrupted — which is exactly why hook
  events survive HPC job kills while claude_stdout.txt does not (the runner writes that only after subprocess.run() returns).

  The role of hook_events.jsonl is twofold:
  1. Tracing: it gives you a turn-by-turn record of what tools Claude called and in what order, independent of whether the session completed
  2. Diagnostics: it's the only artifact that survives a mid-run kill, so it's what we used to figure out that those 26 lv2 tasks were killed by HPC walltime

  ---
  5. The outputs written after the subprocess returns

  Once claude -p exits (normally or via timeout), the runner writes:
  - claude_stdout.txt / claude_stderr.txt — raw subprocess output
  - claude_output.json — the structured JSON Claude emitted (or the error object)
  - claude_session.jsonl — the full session transcript (messages + tool calls)
  - structured_output.json — the extracted final_answer, confidence, brief_explanation fields
  - normalized_trace.json — a runner-normalized summary combining all of the above into a single record (success/failure, token counts, tool call sequence, etc.)

  ---
  Summary pipeline

  gaia_lv1_x4.jsonl (input)
    → task_record.json          [copy of input line]
    → task_prompt.txt           [rendered prompt template]
    → cli_command.json          [full claude -p ... argument list]
    → subprocess launched: claude -p ...
        ↓ (live, via hook script)
        hook_events.jsonl       [per-event log: SessionStart → tool calls → Stop]
        ↓ (after subprocess exits)
        claude_stdout/stderr.txt
        claude_output.json
        claude_session.jsonl
        structured_output.json
        normalized_trace.json

  The split between "written before launch" (task_record, task_prompt, cli_command) and "written after exit" (everything else) is exactly what explains the failure
  patterns we've been debugging: tasks killed mid-run have only the first three files plus partial hook events.