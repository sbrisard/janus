*********************************
Claude's contributions to `Janus`
*********************************

As of august 2026, `Janus` is revived by the author with the help of `Claude Code`. This page will collect all interactions between the author and Claude.


2026-09-11 • How to update this page
====================================

@Claude: this paragraph describe how we are to interact for the development of `Janus`. Take note of these instructions so that I don't have to repeat myself at each new session.

Each time you are invoked in a console, you will read the present file and find the paragraphs that are marked `TODO`.

These are your instructions for the current session: each paragraph is an elementary task. As far as possible, each task will be given its own section with a short, unique title.

After completion of a task, report on what you did in a sub-paragraph called `Claude's report` of the `claude.rst` file.

Once I have validated your changes, I will ask you to mark the task as done. This means you will replace the `TODO` flag with the date in yyyy-mm-dd format. You will then move to the next task. For tasks that cannot be completed in one session, use an `IN PROGRESS` flag, so that the state of the task is visible in the present file.

Unless otherwise stated, you will never commit your changes.

Before you mark the present task as done, you can collect in a sub-paragraph any suggestions that might improve our interactions.

For these instructions to apply on any machine, add the following pointer in `CLAUDE.md`

  at session start, read `sphinx/claude.rst` and process the `TODO` sections according to the `How to update this page` section


Claude's report
---------------

The pointer above was added to `CLAUDE.md`, in a new `Session workflow` section at the top of the file. Since `CLAUDE.md` is loaded at the start of every session, the present file is now the single source of truth for our workflow, on any machine.

The copy of these instructions that had been saved in Claude Code's local memory (under `~/.claude/projects/`, on the author's machine only) was deleted: it duplicated the present file and was already out of date (it did not require the author's validation before marking a task as done).

The author validated these changes on 2026-09-11, and the task was marked as done.
