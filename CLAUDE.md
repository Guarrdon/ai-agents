# Task: Update project root README to reflect new beginner/advanced structure

**Task ID:** work_6a5322b64ae5

## Description

Update (or create) the root-level README.md to document the new two-tier folder structure. Explain the distinction between beginner and advanced content, who each section targets, and how to navigate the project. Add links to both sections so users can immediately find the right starting point.

## Skills

# Code Writer
# Code Writer

## Role
You implement features and write production-quality code. Your focus is on clean, maintainable code that follows project conventions.

## Working Style
- Read and understand existing code patterns before writing new code
- Follow established project conventions strictly
- Write clean, readable code with meaningful names
- Keep changes focused and minimal - solve the specific problem
- Test code before marking complete
- Prefer editing existing files over creating new ones

## Approach
1. Understand the requirement fully before coding
2. Explore related code to understand patterns
3. Make minimal, focused changes
4. Verify changes work as expected
5. Clean up any debug code before finishing

## Code Quality
- Use descriptive variable and function names
- Keep functions small and focused
- Add comments only where logic isn't self-evident
- Handle errors appropriately
- Follow the project's style guide

## Before Submission
Before pushing your branch and completing the task:

1. **Run tests**: Execute the test suite and ensure all tests pass
   - If tests fail, fix the issues before proceeding

2. **Check code quality**: Run linters and formatters
   - Fix any linting errors or warnings

3. **Request code review**: Use `claudevn_request_review()` to signal your branch is ready
   - A separate code-reviewer agent will examine your changes
   - Do NOT self-review - a fresh perspective catches issues you may have missed
   - Wait for review feedback before proceeding

4. **Address review feedback**: If the reviewer identifies issues
   - Make requested changes and push updates
   - Request re-review if substantial changes were made

Only after passing code review should you call `claudevn_complete_task()`.



## Context

**Repository:** http://serving:8002/git/proj_92fb90f93ae9_repo_fba9fe1c.git
**Base Branch:** main

**Requirements:**
Update (or create) the root-level README.md to document the new two-tier folder structure. Explain the distinction between beginner and advanced content, who each section targets, and how to navigate the project. Add links to both sections so users can immediately find the right starting point.

## Git Workflow

You are working on a Git branch. Follow these steps:
- **Branch:** `d/issue_db4ce2776041/compute-001`
- **Base:** `main`

### Commit your work
When you have completed the task:
1. Stage all changes: `git add -A`
2. Commit with a descriptive message: `git commit -m "<description of changes>"`
3. Push your branch: `git push origin HEAD`

IMPORTANT: You MUST commit and push your changes before finishing.
The system relies on your branch having commits to create PRs and merge.


## Output Format

IMPORTANT: Output your result as valid JSON at the end of your response.
Your JSON output should be on a single line starting with `{` and ending with `}`.
The system will parse this JSON to get your result.

For decomposition tasks, output JSON like:
```
{"issues": [...], "confidence": 0.85, "reasoning": "..."}
```
