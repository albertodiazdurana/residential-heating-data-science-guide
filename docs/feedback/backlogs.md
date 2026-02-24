# DSM Backlog Feedback

Track proposed improvements to DSM discovered during this project.

| Date | DSM Section | Issue/Gap | Proposed Improvement | Priority |
|------|-------------|-----------|---------------------|----------|
| 2026-02-19 | dsm-go Step 6 | Baseline script fails on deleted files | Filter deleted entries before md5sum | Medium |

### [2026-02-19] dsm-go Step 6: baseline script fails when working tree has deleted files

**Type:** Backlog Proposal
**Priority:** Medium
**Source:** dsm-residential-heating-ds-guide

**Problem:** The session baseline script in `/dsm-go` Step 6 runs `git status --porcelain | grep -v '^\?' | awk '{print $2}' | xargs -r md5sum` to checksum tracked files. When a file has been deleted from the working tree (git status shows `D` prefix), `md5sum` fails because the file does not exist on disk.

**Evidence:** Session 1 of dsm-residential-heating-ds-guide. `05_Part_V_Interview_Scenarios.md` was deleted (renamed to `05_Part_V_Applied_Scenarios.md`). The baseline script exited with code 123 (`md5sum: 05_Part_V_Interview_Scenarios.md: No such file or directory`).

**Proposed Solution:** Add a filter to exclude deleted files before the checksum step. Replace:
```bash
git status --porcelain | grep -v '^\?' | awk '{print $2}' | xargs -r md5sum
```
With:
```bash
git status --porcelain | grep -v '^\?' | grep -v '^.D' | awk '{print $2}' | xargs -r md5sum
```

**Pushed:** 2026-02-19
