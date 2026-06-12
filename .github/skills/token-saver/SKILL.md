---
name: token-saver
description: 'Compress workspace context before handing a task to an agent. Use for large projects, multi-file refactors, broad code reviews, semantic context compression, token reduction, pruning irrelevant DTOs, excluding tests, dropping long comments, and minimizing prompt size while keeping behavior-critical code paths.'
argument-hint: 'Describe the task, target files or symbols, and any files that must be preserved in context'
user-invocable: true
disable-model-invocation: false
---

# Token Saver

Use this skill when the workspace is large enough that naive context gathering would waste tokens, especially during multi-file refactors, architecture tracing, broad debugging, or agent handoff.

The goal is to preserve only behavior-critical context for the current task and aggressively compress or exclude everything else.

## When to Use
- Large repositories with many unrelated modules
- Multi-file refactors where only a few control paths matter
- Requests that mention agents, broad exploration, or handoff
- Reviews or explanations of long files where full-file inclusion is unnecessary
- Situations where DTO definitions, tests, generated files, or long comments dominate context without affecting the immediate task

## Do Not Use
- Small, single-file edits where compression costs more than it saves
- Tasks centered on tests, contract objects, schemas, or generated interfaces that are themselves the subject of the change
- Cases where comments contain required design constraints, safety requirements, migration notes, or protocol semantics

## Outcome
Produce a compressed working set that:
- Keeps the owning abstraction, the active control path, and the minimum neighboring call sites needed to act
- Removes or summarizes irrelevant DTOs, tests, fixtures, generated code, and verbose comments
- Preserves names, signatures, invariants, and task-critical behavior
- Minimizes the token footprint before invoking another agent or performing broad reasoning

## Operating Rules
- Be concise by default
- Prefer code excerpts, symbol lists, and summaries over full-file inclusion
- Reuse existing workspace libraries and patterns; do not introduce new dependencies just to inspect or compress context
- When explaining long files, produce a short structural summary unless the task requires line-level reasoning
- Default internal constraint: code only, no explanation, unless the task is explicitly conceptual or a non-obvious bug requires explanation
- Never drop context that defines correctness: public interfaces, behavior-affecting config, safety checks, validation, serialization boundaries, or migration-sensitive assumptions

## Procedure
1. Identify the immediate task anchor.
Choose the most concrete starting point available: named file, symbol, failing command, failing test, or user-specified behavior.

2. Build a minimal relevance graph.
Keep only:
- The owning implementation
- Direct callers or callees needed to understand control flow
- Types or config that affect runtime behavior for this task
- One nearby test only if it is the cheapest discriminating check

3. Classify surrounding material.
For each nearby file or block, label it as one of:
- Keep: directly needed for behavior or editing
- Summarize: useful context but too verbose to inline
- Drop: unrelated to the active task

4. Apply semantic compression.
Compress in this order:
- Replace long unchanged regions with short summaries
- Omit DTOs or data containers that only mirror fields and are not being changed
- Omit tests, fixtures, snapshots, and mocks unless they validate the current slice
- Remove banner comments, repeated docstrings, and long prose comments that do not change behavior
- Collapse generated or repetitive code to signatures, field names, and one-line purpose notes

5. Preserve critical invariants.
Before handoff, explicitly retain:
- File paths and owning symbols
- Function signatures that may be edited
- Preconditions, postconditions, and validation logic
- Config flags or constants that alter behavior
- Any user constraints about style, API stability, or forbidden edits

6. Prepare the handoff packet.
Pass only:
- Task summary in 1 to 3 lines
- Target files and symbols
- Compressed behavior summary
- Open question or hypothesis, if one exists
- Focused validation target such as a specific test, typecheck, or command

7. Validate compression quality.
Ask whether the reduced context still lets another agent answer these questions:
- What code actually controls the behavior?
- What is safe to edit?
- What check will prove the change worked?
If any answer is unclear, restore only the missing local context.

## Decision Points
- If the user names exact files or symbols, start there and avoid broad search.
- If a file mostly contains wiring, step one hop toward the code that computes or mutates behavior.
- If tests are large and indirect, drop them and keep only the smallest validating test or command.
- If DTOs include validation or serialization behavior, keep the relevant members instead of dropping the file wholesale.
- If comments encode hidden constraints, summarize the constraint instead of deleting it.
- If multiple paths look plausible, keep the path with the cheapest falsifiable check and drop the others.

## Compression Heuristics
- Prefer symbol names over full bodies when bodies are repetitive
- Prefer signatures plus 1 to 3 behavior bullets over full class dumps
- Prefer one representative example over repeated patterns
- Prefer targeted excerpts over entire files
- Exclude files under common test, fixture, mock, snapshot, build, dist, and generated paths unless directly relevant

## Completion Check
The skill is complete when:
- The active task can be described without restating unrelated repository details
- The retained context identifies the controlling code path and safe edit surface
- Token-heavy but irrelevant content has been removed or summarized
- A downstream agent could act without reopening broad exploration
- A concrete validation step is included with the compressed context

## Example Prompts
- /token-saver Compress context for refactoring the reward pipeline under source/b2_z1 and keep only behavior-critical files
- /token-saver Prepare an agent handoff for a multi-file rename around pose commands; exclude unrelated DTOs, tests, and long comments
- /token-saver Summarize the control path for this bug and prune everything not needed for the next edit

## Notes
Use this as a pre-processing workflow before invoking Explore or another agent on large tasks. Prefer restoring one missing local dependency over reloading whole files.