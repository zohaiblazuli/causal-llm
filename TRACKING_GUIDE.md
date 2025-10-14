# Project Tracking Guide

## Purpose

This guide explains how to use the project tracking infrastructure to maintain visibility and coordination throughout the 6-month ISEF research project.

---

## Document Structure

### Core Tracking Documents

1. **PROJECT_STATUS.md** - The single source of truth for overall project status
   - Update: After completing each milestone
   - Contains: Current phase, milestone status, blockers, decisions, file references
   - Audience: Anyone needing high-level project understanding

2. **MILESTONES.md** - Detailed breakdown of all phases and milestones
   - Update: Rarely (only if scope changes)
   - Contains: Success criteria, dependencies, risk assessments
   - Audience: Detailed planning and reference

3. **WEEKLY_LOG_TEMPLATE.md** - Template for weekly progress reports
   - Update: Every week (copy template, fill in, save as WEEKLY_LOG_YYYY_MM_DD.md)
   - Contains: Progress summary, completed tasks, upcoming priorities, blockers
   - Audience: Weekly reviews and retrospectives

4. **DAILY_STANDUP_TEMPLATE.md** - Template for daily progress tracking
   - Update: Daily (copy template, fill in, save as DAILY_YYYYMMDD.md)
   - Contains: Yesterday's work, today's plan, blockers
   - Audience: Personal tracking and daily coordination

---

## Update Workflow

### Daily Routine

**Morning (5-10 minutes):**
1. Copy DAILY_STANDUP_TEMPLATE.md to logs/DAILY_YYYYMMDD.md
2. Fill in today's plan and priorities
3. Review any blockers from yesterday
4. Check upcoming milestone deadlines

**Evening (5-10 minutes):**
1. Update daily standup with accomplishments
2. Check off completed tasks
3. Note any new blockers or issues
4. Prepare tomorrow's priorities

### Weekly Routine

**End of Week (30-45 minutes):**
1. Copy WEEKLY_LOG_TEMPLATE.md to logs/WEEKLY_LOG_WEEK_XX.md
2. Summarize week's accomplishments
3. Update milestone status in PROJECT_STATUS.md
4. Review and update blocker tracking
5. Plan next week's priorities
6. Update progress metrics

### Milestone Completion

**When Milestone Achieved (15-30 minutes):**
1. Mark milestone as COMPLETED in PROJECT_STATUS.md
2. Update phase completion percentage
3. Document key decisions made
4. Add deliverables to file references
5. Review next milestone's dependencies
6. Update risk register if needed

### Phase Transitions

**When Phase Completes (1-2 hours):**
1. Complete comprehensive phase retrospective
2. Update PROJECT_STATUS.md phase status table
3. Document all deliverables and outcomes
4. Review MILESTONES.md for next phase
5. Update risk assessment for upcoming phase
6. Create phase completion summary
7. Plan kickoff for next phase

---

## File Organization

### Recommended Directory Structure

```
c:\isef\
├── PROJECT_STATUS.md              # Main status document
├── MILESTONES.md                  # Detailed milestone definitions
├── TRACKING_GUIDE.md              # This file
├── WEEKLY_LOG_TEMPLATE.md         # Weekly template
├── DAILY_STANDUP_TEMPLATE.md      # Daily template
├── logs/                          # All logs stored here
│   ├── daily/
│   │   ├── DAILY_20241201.md
│   │   ├── DAILY_20241202.md
│   │   └── ...
│   ├── weekly/
│   │   ├── WEEKLY_LOG_WEEK_01.md
│   │   ├── WEEKLY_LOG_WEEK_02.md
│   │   └── ...
│   └── phase_retrospectives/
│       ├── PHASE_1_RETROSPECTIVE.md
│       └── ...
├── deliverables/                  # Phase deliverables
│   ├── phase1/
│   ├── phase2/
│   └── ...
└── [other project files]
```

---

## Progress Tracking Best Practices

### Status Updates

**Be Specific:**
- BAD: "Made progress on causal model"
- GOOD: "Completed formal definition of intervention operator (Section 2.3 of causal_model.pdf)"

**Quantify When Possible:**
- BAD: "Working on dataset"
- GOOD: "Generated 3,247 / 10,000 target contrastive pairs (32% complete)"

**Distinguish States Clearly:**
- NOT STARTED: No work has begun
- IN PROGRESS: Actively working, < 90% complete
- BLOCKED: Cannot proceed without resolution
- COMPLETED: Meets all success criteria, reviewed and validated

### Blocker Management

**Identify Early:**
- Don't wait for a blocker to completely stop progress
- Flag potential blockers as "At Risk"

**Categorize Clearly:**
- CRITICAL: Stops all progress on phase
- HIGH: Blocks milestone, impacts timeline
- MEDIUM: Slows progress, workarounds exist
- LOW: Minor friction, minimal impact

**Actionable Mitigation:**
- Every blocker needs a mitigation plan
- Assign ownership (even if self)
- Set target resolution date
- Update status regularly

### Risk Tracking

**Continuous Assessment:**
- Review risk register weekly
- Add new risks as identified
- Update probabilities as project progresses
- Archive resolved risks with notes on how they were handled

**Honest Evaluation:**
- Don't minimize risks to appear on track
- Escalate high-impact risks immediately
- Document mitigation attempts even if unsuccessful

---

## Milestone Completion Criteria

### How to Know a Milestone is Complete

A milestone is only complete when ALL of the following are true:

1. **All tasks checked off** in the milestone task list
2. **Success criteria met** as defined in MILESTONES.md
3. **Deliverables created** and documented in file references
4. **Validation completed** (review, testing, advisor feedback as appropriate)
5. **Documentation updated** to reflect completion
6. **Handoff ready** if next milestone depends on this one

### Avoiding "Almost Done" Syndrome

- 90% complete is NOT complete
- If you discover additional work needed, add it to the task list
- Update completion percentage honestly
- Don't mark complete until independently verifiable

---

## Communication Guidelines

### When to Update PROJECT_STATUS.md

**Always Update When:**
- Completing a milestone
- Discovering a blocker
- Making a key decision
- Phase transition
- Timeline concern emerges

**Consider Updating When:**
- Significant progress on current milestone (>25% increment)
- Risk probability or impact changes
- New dependencies identified
- Deliverable locations change

### When to Escalate

**Escalate Immediately If:**
- Critical blocker with no clear resolution path
- Timeline slippage >10% of phase duration
- Scope change needed to meet deadlines
- Resource constraints threaten project success
- Strategic decisions required beyond your authority

**Escalation Process:**
1. Document the issue clearly in PROJECT_STATUS.md
2. Assess impact and propose options
3. Schedule meeting with advisor/stakeholder
4. Document decision and rationale

---

## Context Window Continuity

### Designing for Context Loss

These documents are designed so that if you lose context (new conversation, different agent, etc.), someone can:

1. Read PROJECT_STATUS.md to understand current state
2. Read MILESTONES.md to understand success criteria
3. Read recent weekly logs to understand recent progress
4. Continue work without loss of momentum

### Critical Information Always Visible

**PROJECT_STATUS.md should always show:**
- What phase we're in
- What milestone is active
- Current blockers
- Recent key decisions
- Where to find important files

**If context is lost, start here:**
1. Open PROJECT_STATUS.md
2. Check "Current Phase" section
3. Review active milestones and their status
4. Check blocker section
5. Review last weekly log
6. Identify next priorities

---

## Tools & Automation

### Suggested Tools

**Document Management:**
- Version control (Git) for all tracking documents
- Markdown editor with preview
- File search tool for finding references

**Time Tracking:**
- Simple time tracker for accurate hour logging
- Calendar for milestone deadlines
- Reminders for weekly/daily updates

**Communication:**
- Shared calendar for advisor meetings
- Email/Slack for blocker escalation
- Screen recording for demos and presentations

### Automation Opportunities

**Can Be Automated:**
- Copying daily/weekly templates with correct dates
- Calculating days remaining in phase
- Checking for overdue milestones
- Generating progress graphs from logs

**Should Stay Manual:**
- Writing status summaries
- Assessing risks and blockers
- Making decisions about priorities
- Evaluating milestone completion

---

## Troubleshooting

### "I don't know what to work on next"

1. Check PROJECT_STATUS.md "Next Actions" section
2. Review current milestone in MILESTONES.md
3. Look at dependencies - what's blocking downstream work?
4. Check weekly log for planned priorities
5. If still unclear, escalate for guidance

### "I'm behind schedule"

1. Quantify the delay (days/weeks)
2. Identify root cause (complexity, blocker, scope creep)
3. Document in blocker section with HIGH severity
4. Propose mitigation options:
   - Reduce scope of current milestone
   - Extend phase timeline (impact on overall project)
   - Increase effort/hours
   - Seek help/resources
5. Escalate if delay impacts critical path

### "A milestone's success criteria is unclear"

1. Review MILESTONES.md detailed description
2. Check if there are example deliverables from similar projects
3. Draft your interpretation and document it
4. Validate with advisor before significant work
5. Update MILESTONES.md with clarification for future reference

### "I completed a milestone but not sure what's next"

1. Mark current milestone COMPLETED in PROJECT_STATUS.md
2. Check MILESTONES.md for dependencies
3. Identify next milestone in sequence
4. Review its success criteria and tasks
5. Update PROJECT_STATUS.md with new active milestone
6. Plan first tasks in next daily standup

---

## Coordinator Role & Responsibilities

### As Project Coordinator, You Should:

**Daily:**
- Monitor progress against milestones
- Identify blockers and flag them
- Ensure one person/agent isn't waiting on another
- Keep file references up to date

**Weekly:**
- Generate comprehensive weekly reports
- Review timeline adherence
- Update risk register
- Facilitate coordination between workstreams
- Celebrate completed milestones

**Phase Transitions:**
- Conduct retrospectives
- Validate all deliverables
- Ensure handoff is smooth
- Update overall project timeline
- Brief team on next phase

### Coordination Principles

1. **Proactive, Not Reactive:** Anticipate needs before they're urgent
2. **Single Source of Truth:** PROJECT_STATUS.md is authoritative
3. **Outcome-Focused:** Care about impact, not just activity
4. **Clear Communication:** Be concise but complete
5. **Honest Assessment:** Truth over optimism

---

## Quick Reference

### Update Frequency Cheat Sheet

| Document | Update Frequency | Time Required |
|----------|-----------------|---------------|
| DAILY_STANDUP | Every workday | 5-10 min |
| WEEKLY_LOG | End of each week | 30-45 min |
| PROJECT_STATUS | After each milestone | 15-30 min |
| MILESTONES | Rarely (scope changes only) | As needed |

### Status Definitions

- **ON TRACK:** Meeting all milestones within timeline
- **AT RISK:** Behind by <1 week or facing potential blocker
- **BEHIND:** Behind by >1 week, mitigation plan needed
- **AHEAD:** Completed early, can advance to next milestone

### Priority Levels

- **CRITICAL:** Must be done today, blocks everything
- **HIGH:** Must be done this week, blocks milestone
- **MEDIUM:** Should be done soon, good to complete
- **LOW:** Nice to have, can be deferred if needed

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-12 | Initial tracking guide created |

---

**Remember:** The purpose of tracking is not bureaucracy - it's to ensure you complete a successful 6-month research project on time. These tools should help you, not slow you down. Adapt as needed, but always maintain visibility into progress, blockers, and next steps.