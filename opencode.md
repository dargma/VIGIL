Claude Code Instruction: Generate OpenCode/OmO-Ready Branch

You are refactoring this project for the OpenCode + Oh-My-OpenCode (OmO) multi-agent environment.
Create a new git branch `opencode` and generate all necessary configuration files so that running `opencode` in this branch immediately works with full OmO orchestration.

---

## STEP 0: Understand the Project

Read the entire project first. In order:

1. Read `CLAUDE.md` (or any existing instruction files)
2. Read `SKILL.md`, `HANDOFF.md`, `STATUS.md`, `RESEARCH_JOURNAL.md` if they exist
3. Scan the full directory tree
4. Read key source files to understand architecture
5. Read any experiment logs, configs, or result files
6. Read any existing `requirements.txt`, `setup.py`, `pyproject.toml`

From this, extract:
- **Mission**: What is this project trying to achieve?
- **Current state**: What has been done? What are the latest results?
- **Theory**: What is the theoretical framework?
- **Technical details**: Key constants, dimensions, model names, dataset paths, known pitfalls
- **Experiment history**: Past experiments and their outcomes
- **Decision log**: Why certain approaches were chosen
- **Paper references**: Any referenced papers or literature

Do NOT proceed until you have a thorough understanding. Use `ultrathink` for this.

---

## STEP 1: Create Branch

```bash
git checkout -b opencode
```

---

## STEP 2: Generate `AGENTS.md`

Create `AGENTS.md` at project root. This replaces CLAUDE.md and is auto-injected into every agent prompt.

Requirements:
- Extract mission, current phase, architecture from existing docs
- Include the following sections (adapt content to THIS project):

```
# [Project Name]

## Mission
[Extract from existing CLAUDE.md / README]

## Current Phase
[Extract from HANDOFF.md / STATUS.md / RESEARCH_JOURNAL.md / latest experiment]

## Agent Role
You are an autonomous researcher, not a code executor.
Design experiments → run them → interpret results → decide next direction.
Make decisions based on data and theory. If ambiguous, pick the simpler option.

## Decision Authority
FULL AUTONOMOUS AUTHORITY for all operations.
FORBIDDEN: "Should I...", "Do you want me to...", "Shall I..."
Only stop for: unrecoverable GPU OOM, disk full, or authentication needed.

## Parallel Execution (MANDATORY)
- GPU training → spawn Librarian for paper analysis (run_in_background=true)
- Evaluation → prepare next experiment code (run_in_background=true)
- NEVER wait idle.

## Experiment Protocol
1. Read docs/EXPERIMENT_LOG.md → current state
2. Design: hypothesis + ONE variable change + expected outcome
3. Execute (background agents for long tasks)
4. Record in docs/EXPERIMENT_LOG.md (including failures)
5. 3 consecutive no-improvement → change axis
6. Git commit + push at every milestone

## Session Continuity
- Start: read docs/EXPERIMENT_LOG.md
- During: update after each experiment
- Periodically: git commit + push

## Environment
[Extract GPU specs, paths, constraints from existing docs]
Semi-airgapped: arXiv PDF download OK, active web search blocked.

## Paper Protocol
- Store PDFs in papers/
- Read: pdftotext papers/filename.pdf -
- Maintain papers/index.md
- New paper: wget from arXiv → /paper-review

## Architecture
[Extract from existing code — draw text diagram of modules and data flow]

## Key Technical Details
[Extract ALL critical implementation details, constants, known pitfalls from existing code and docs]

## Do NOT Touch
[List protected files if any]
```

Keep AGENTS.md **concise** — detailed content goes into docs/ files loaded via instructions.

---

## STEP 3: Generate `opencode.json`

Create `opencode.json` at project root:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "anthropic/claude-sonnet-4-20250514",
  "instructions": [
    "docs/THEORY.md",
    "docs/EXPERIMENT_LOG.md",
    "docs/DECISIONS.md",
    "papers/index.md"
  ],
  "permission": {
    "edit": "allow",
    "bash": {
      "python *": "allow",
      "pip *": "allow",
      "git *": "allow",
      "nvidia-smi": "allow",
      "torchrun *": "allow",
      "accelerate *": "allow",
      "wget *": "allow",
      "pdftotext *": "allow",
      "*": "allow"
    },
    "webfetch": "allow",
    "skill": { "*": "allow" }
  },
  "mcp": {}
}
```

Adjust the `instructions` list to include only files that actually exist or that you will create.

---

## STEP 4: Generate `.opencode/oh-my-opencode.jsonc`

```bash
mkdir -p .opencode
```

Create `.opencode/oh-my-opencode.jsonc`:

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/master/assets/oh-my-opencode.schema.json",
  "disabled_mcps": ["websearch", "context7", "grep_app"],
  "agents": {
    "sisyphus": {
      "temperature": 0.2,
      "skills": ["controlled-experiment"]
    },
    "multimodal-looker": { "disable": true }
  },
  "categories": {
    "ml-experiment": {
      "description": "ML experiment execution",
      "temperature": 0.1,
      "prompt_append": "Controlled experiment protocol. One variable. Record everything. Check GPU memory."
    },
    "paper-analysis": {
      "description": "Paper PDF analysis and literature review",
      "temperature": 0.3,
      "prompt_append": "Read from papers/ via pdftotext. Cross-reference THEORY.md. Update papers/index.md."
    },
    "eval-analysis": {
      "description": "Experiment result analysis and next experiment proposal",
      "temperature": 0.2,
      "prompt_append": "Compare against EXPERIMENT_LOG.md. Identify trends. Propose next experiment with hypothesis."
    },
    "debug-investigate": {
      "description": "Training failure analysis and debugging",
      "temperature": 0.1,
      "prompt_append": "Systematic debugging. NaN→rollback+LR/10. Collapse→variance loss. OOM→reduce batch."
    }
  },
  "ralph_loop": { "enabled": true },
  "background_task": {
    "defaultConcurrency": 5,
    "providerConcurrency": { "anthropic": 3, "openai": 5, "google": 10 }
  },
  "disabled_hooks": ["auto-update-checker"],
  "claude_code": {
    "mcp": false,
    "commands": true,
    "skills": true,
    "hooks": false,
    "plugins": true
  }
}
```

**Project-specific adjustments**: Add project-specific categories or modify `prompt_append` based on the project's domain. For example:
- Alpha-Triton: add `"prompt_append": "... JEPA only, no contrastive. 4800+ pairs. Cache cleanup between batches."`
- VLM GRPO: add `"prompt_append": "... remove_unused_columns=False. Separate MME cognition vs perception."`

---

## STEP 5: Generate Skills

```bash
mkdir -p .opencode/skills/controlled-experiment
mkdir -p .opencode/skills/ml-researcher
```

### .opencode/skills/controlled-experiment/SKILL.md

```markdown
---
name: controlled-experiment
description: Controlled experiment protocol — one variable, systematic recording, decision rules for axis switching
---

# Controlled Experiment Protocol

## Rules
1. Change ONE variable at a time
2. Every experiment: hypothesis + expected outcome + actual outcome
3. Record ALL experiments (including failures) in docs/EXPERIMENT_LOG.md
4. 3 consecutive no-improvement → change axis (Architecture → Loss → Hyperparameter)
5. Quick eval (<5min) before full eval when possible

## Record Format
experiment_id: exp_NNN
hypothesis: [what and why]
changes: [exactly what changed]
config: [reference]
results: [metrics]
analysis: [why this happened]
next: [proposal]

## Decision Rules
- NaN → rollback + LR/10
- Collapse (std < 0.01) → rollback + variance loss
- OOM → reduce batch_size first, then num_generations, then max_length
- Disk < 5GB → purge caches
- 3× no improvement → switch axis
```

### .opencode/skills/ml-researcher/SKILL.md

```markdown
---
name: ml-researcher
description: Autonomous ML research protocol — parallel execution, paper analysis, experiment design
---

# ML Researcher Protocol

## Philosophy
Mission + Theory + you figure out execution.
You receive research goals and theory. You design experiments and implementation.

## Parallel Execution
- GPU busy → background paper analysis / code prep
- NEVER idle

## Paper Analysis (semi-airgapped)
- Read: pdftotext papers/filename.pdf -
- Summarize: motivation, method, results, relevance
- Update papers/index.md

## Experiment Design
- One variable at a time
- Hypothesis before execution
- Success/failure criteria upfront
- Record everything
```

### Project-Specific Skill

**Create ONE additional skill** specific to this project. Extract ALL critical implementation details, constants, known pitfalls, and domain knowledge from the existing codebase and docs.

Name it based on the project (e.g., `vlm-grpo`, `alpha-triton`, `mrtg-analysis`, `geneeg`).

The skill MUST include:
- Key constants and dimensions
- Critical configuration requirements (e.g., `remove_unused_columns=False`)
- Known pitfalls discovered during past experiments
- Evaluation specifics (which benchmarks, how to report)
- Any domain-specific patterns or anti-patterns

---

## STEP 6: Generate Commands

```bash
mkdir -p .opencode/commands
```

### .opencode/commands/run-experiment.md
```markdown
---
description: Execute next experiment from the experiment log
agent: sisyphus
---
Execute the next experiment:
1. Read docs/EXPERIMENT_LOG.md — find last experiment and "Next" entry
2. Implement code changes
3. Check GPU memory before launching
4. Run training
5. Record results in docs/EXPERIMENT_LOG.md
6. Analyze and propose next experiment
7. Git commit + push
```

### .opencode/commands/eval.md
```markdown
---
description: Run benchmark evaluation on current checkpoint
agent: sisyphus
---
Evaluate the current checkpoint:
1. Run evaluation script or benchmark tool
2. Compare against previous results in docs/EXPERIMENT_LOG.md
3. Report improvement/regression with analysis
```

### .opencode/commands/paper-review.md
```markdown
---
description: Analyze a paper PDF and update the paper index
subtask: true
---
Analyze: papers/$1.pdf
1. pdftotext papers/$1.pdf -
2. Extract: motivation, method, key results, limitations
3. Relevance to our project (reference docs/THEORY.md)
4. If applicable → propose experiment
5. Update papers/index.md
```

### .opencode/commands/status-report.md
```markdown
---
description: Generate concise project status
subtask: true
---
Status report:
1. Last 3 experiments from docs/EXPERIMENT_LOG.md
2. Current best metrics
3. Active hypotheses
4. Blockers
5. Next steps (prioritized)
Max 20 lines.
```

---

## STEP 7: Generate docs/

```bash
mkdir -p docs papers
```

### docs/THEORY.md
Extract the theoretical background of this project from existing docs/code/comments.
Include: core hypothesis, mathematical formulation if any, key references, design rationale.

### docs/EXPERIMENT_LOG.md
Consolidate ALL past experiment results from:
- RESEARCH_JOURNAL.md
- lab/registry/
- Any logs, metrics, or reports found in the project

Format each entry as:
```
## exp_NNN: [Name]
- Date: YYYY-MM-DD
- Hypothesis: ...
- Changes: ...
- Results: ...
- Analysis: ...
- Next: ...
```

If no experiment history exists, create a template with `exp_001: Baseline` as placeholder.

### docs/DECISIONS.md
Extract design decisions and their rationale from existing docs and code comments.
Format: "We chose X over Y because Z."

### docs/REPRODUCTION.md
Write a reproduction guide:
- Environment setup (packages, GPU requirements)
- Data preparation steps
- How to run training
- How to run evaluation
- Expected results

### papers/index.md
If papers are referenced in the project, create an index. Otherwise create an empty template:
```markdown
# Paper Index
| Filename | Title | Key Technique | Relevance |
|----------|-------|--------------|-----------|
```

---

## STEP 8: Clean Up Old Files

Do NOT delete old files. Instead, add a note at the top of each deprecated file:

```markdown
<!-- DEPRECATED: Migrated to OpenCode/OmO format. See AGENTS.md + opencode.json + .opencode/ -->
```

Apply this to:
- CLAUDE.md (if exists)
- SKILL.md (root-level, if exists)
- HANDOFF.md (if exists)
- STATUS.md (if exists)

---

## STEP 9: Verify Structure

Run this check and fix any issues:

```bash
echo "=== Verification ==="
echo "AGENTS.md:        $(test -f AGENTS.md && echo OK || echo MISSING)"
echo "opencode.json:    $(test -f opencode.json && echo OK || echo MISSING)"
echo "oh-my-opencode:   $(test -f .opencode/oh-my-opencode.jsonc && echo OK || echo MISSING)"
echo ""
echo "Skills:"
find .opencode/skills -name "SKILL.md" 2>/dev/null | while read f; do echo "  $f"; done
echo ""
echo "Commands:"
find .opencode/commands -name "*.md" 2>/dev/null | while read f; do echo "  $f"; done
echo ""
echo "Docs:"
for f in docs/THEORY.md docs/EXPERIMENT_LOG.md docs/DECISIONS.md docs/REPRODUCTION.md papers/index.md; do
  echo "  $f: $(test -f $f && echo OK || echo MISSING)"
done
echo ""
echo "JSON validation:"
python3 -c "import json; json.load(open('opencode.json')); print('  opencode.json: valid')" 2>/dev/null || echo "  opencode.json: INVALID"
echo ""
echo "Skill name validation:"
find .opencode/skills -name "SKILL.md" -exec grep -l "^name:" {} \; | while read f; do
  name=$(grep "^name:" "$f" | head -1 | sed 's/name: *//')
  dir=$(basename $(dirname "$f"))
  if [ "$name" = "$dir" ]; then
    echo "  $name: OK (matches dir)"
  else
    echo "  $name: MISMATCH (dir=$dir)"
  fi
done
```

---

## STEP 10: Commit and Report

```bash
git add -A
git commit -m "feat: migrate to OpenCode/OmO multi-agent research environment

- AGENTS.md: project instructions (replaces CLAUDE.md)
- opencode.json: model, permissions, instruction loading
- .opencode/oh-my-opencode.jsonc: OmO agent/category/Ralph Loop config
- .opencode/skills/: controlled-experiment, ml-researcher, [project-specific]
- .opencode/commands/: run-experiment, eval, paper-review, status-report
- docs/: THEORY, EXPERIMENT_LOG, DECISIONS, REPRODUCTION
- papers/index.md: paper catalog template

Semi-airgapped: web MCPs disabled, local paper analysis via pdftotext.
Ralph Loop enabled for autonomous experiment iteration."
```

Then report:
1. List of all created files
2. Key content extracted from old docs → where it went
3. Any information that could NOT be migrated (and why)
4. Suggested first command to run in OpenCode: `/status-report`
