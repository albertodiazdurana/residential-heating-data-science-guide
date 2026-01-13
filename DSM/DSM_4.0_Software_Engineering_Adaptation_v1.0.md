# DSM 4.0: Software Engineering Adaptation

**Version:** 1.0
**Date:** January 2026
**Purpose:** Extend DSM methodology for ML/software engineering projects where the primary deliverable is a working application, not analytical insights.

---

## 1. When to Use This Adaptation

**Use this adaptation when:**
- Building a software application that uses ML components
- Primary deliverable is code/package, not insights/recommendations
- Output is a tool others will use, not a report stakeholders will read
- Project requires software architecture decisions, not analytical decisions

**Examples:**
- LLM-powered applications (chatbots, report generators, agents)
- ML pipelines and tools
- Data processing applications
- API services with ML backends

**Continue using standard DSM (Sections 1-3) when:**
- Primary goal is data analysis and insights
- Deliverables are notebooks, presentations, reports
- Stakeholder communication of findings is the end goal

---

## 2. Adapted Phase Structure

### Standard DSM Phases (Data Science Projects)
```
Phase 0: Environment Setup
Phase 1: Exploration (EDA, cohort definition)
Phase 2: Feature Engineering
Phase 3: Analysis/Modeling
Phase 4: Communication (reports, presentations)
```

### Adapted Phases (Software Engineering Projects)
```
Phase 0: Environment Setup (unchanged)
Phase 1: Data Pipeline (load, validate, transform)
Phase 2: Core Modules (models, services, providers)
Phase 3: Integration & Evaluation (agents, testing, metrics)
Phase 4: Application & Documentation (UI, README, demos)
```

### Phase Mapping

| Adapted Phase | Focus | Key Activities | Deliverables |
|---------------|-------|----------------|--------------|
| **Phase 0** | Environment | venv, requirements, project structure | Working dev environment |
| **Phase 1** | Data Pipeline | Data loading, validation, PM4Py/pandas integration | `process_analyzer.py`, sample data |
| **Phase 2** | Core Modules | Data models, service classes, provider factories | `models.py`, `llm_provider.py`, `llm_reporter.py` |
| **Phase 3** | Integration | Agent orchestration, evaluation pipeline, testing | Agent module, MLflow integration, tests |
| **Phase 4** | Application | Streamlit app, documentation, demo | `app.py`, README, architecture docs |

---

## 3. Development Protocol

### 3.1 Module Development (replaces Notebook Protocol)

When building application modules:

1. **Explain purpose** before writing code
2. **Build incrementally:** imports → constants → one function → test → next function
3. **Wait for confirmation** before proceeding to next component
4. **User creates all files** - provide code segments to copy/paste
5. **Test-Driven Development:** write tests in `tests/` alongside code

**Interaction pattern:**
- Claude explains purpose → provides code → user creates file → user confirms → next step
- "Done" or "next" = proceed to next step
- "Explain more" = deeper explanation before proceeding

### 3.2 When to Use Notebooks

Notebooks are appropriate for:
- **Exploration:** Understanding data structure, testing PM4Py functions
- **Demos:** `notebooks/01_demo.ipynb` showing end-to-end flow
- **Prototyping:** Quick experiments before committing to module design

Notebooks are NOT for:
- Production code (use `src/` modules)
- Core application logic
- Code that will be imported elsewhere

### 3.3 Code Organization

```
project/
├── src/
│   ├── __init__.py
│   ├── models.py          # Data classes, Pydantic models
│   ├── process_analyzer.py # PM4Py wrapper
│   ├── llm_provider.py    # Provider factory
│   ├── llm_reporter.py    # LangChain integration
│   └── agent.py           # ReAct agent (if applicable)
├── tests/
│   ├── test_models.py
│   ├── test_analyzer.py
│   └── test_provider.py
├── prompts/
│   └── *.txt              # Prompt templates
├── notebooks/
│   └── 01_demo.ipynb      # Demo/exploration only
├── data/
│   └── sample/            # Sample datasets
├── app.py                 # Streamlit application
├── requirements.txt
└── README.md
```

---

## 4. Decision Log Adaptation

### Standard DSM Decision Log (Analytical)
Focuses on: feature selection, model choice, hyperparameters, cohort definition

### Adapted Decision Log (Architectural)
Focuses on: design patterns, library choices, API design, trade-offs

**Template:**

```markdown
## DEC-XXX: [Decision Title]

**Category:** Architecture | Library Choice | API Design | Data Model

**Context:**
[What problem or choice prompted this decision?]

**Decision:**
[What was decided?]

**Alternatives Considered:**
1. [Alternative 1] - Rejected because [reason]
2. [Alternative 2] - Rejected because [reason]

**Rationale:**
[Why this choice?]

**Implications:**
- [What this enables]
- [What this constrains]
- [Future considerations]
```

**Example:**

```markdown
## DEC-001: Provider-Agnostic LLM Factory

**Category:** Architecture

**Context:**
Need to support multiple LLM providers (Anthropic, OpenAI, Ollama) with consistent interface.

**Decision:**
Factory pattern with `AVAILABLE_MODELS` registry and `create_llm()` function.

**Alternatives Considered:**
1. Direct instantiation per provider - Rejected: code duplication, harder to switch
2. Dependency injection container - Rejected: over-engineering for project scope
3. Abstract base class with implementations - Rejected: more boilerplate than needed

**Rationale:**
Factory pattern provides simple, extensible design. Adding new providers requires only:
1. Add entry to AVAILABLE_MODELS
2. Implement _create_[provider]() function

**Implications:**
- Enables runtime provider switching via config
- All providers must conform to LangChain BaseChatModel interface
- Cost tracking possible via ModelConfig dataclass
```

---

## 5. Success Criteria Adaptation

### Standard DSM Success Criteria (Analytical)
- Statistical validity of findings
- Business interpretability
- Stakeholder acceptance
- Reproducible analysis

### Adapted Success Criteria (Software Engineering)

**Functional:**
- [ ] End-to-end pipeline works (data → processing → output)
- [ ] All core features implemented per requirements
- [ ] Error handling for common failure modes

**Code Quality:**
- [ ] Modular, well-organized codebase
- [ ] Functions/classes have docstrings
- [ ] No hardcoded secrets or paths
- [ ] Type hints on public interfaces

**Testing:**
- [ ] Unit tests for core modules
- [ ] Integration test for full pipeline
- [ ] Tests pass in CI (if applicable)

**Documentation:**
- [ ] README with setup instructions
- [ ] Architecture overview
- [ ] API/usage examples
- [ ] Decision log for key choices

**Demo:**
- [ ] Working interactive demo (Streamlit/CLI)
- [ ] Sample data included
- [ ] Demo runs without external dependencies (or documents them)

---

## 6. Portfolio Project Checklist

For projects intended to demonstrate skills (job applications, portfolio):

### Repository Quality
- [ ] Clear, descriptive README
- [ ] Architecture diagram
- [ ] Setup instructions that work
- [ ] Sample data or instructions to obtain it
- [ ] License file

### Code Demonstrates
- [ ] Clean separation of concerns
- [ ] Design patterns where appropriate
- [ ] Error handling
- [ ] Configuration management (env vars, not hardcoded)
- [ ] Logging (not print statements in production code)

### ML/AI Specific
- [ ] Reproducible experiments (seed, logging)
- [ ] Evaluation metrics defined and tracked
- [ ] Model/prompt versioning
- [ ] Clear distinction between training and inference (if applicable)

### JetBrains-Specific (for ML Engineer role)
- [ ] Demonstrates ML system design
- [ ] Shows ability to apply existing models
- [ ] Includes evaluation pipeline
- [ ] Code is in Python (their primary language for ML)
- [ ] Modern frameworks (LangChain, HuggingFace, PyTorch)
- [ ] Agentic patterns (if applicable)

---

## 7. Sprint Planning for SW Projects

### Example: 4-Day ML Application Project

**Day 1: Foundation**
- Phase 0: Environment setup
- Phase 1: Data pipeline
- Deliverable: Working data loading and basic analysis

**Day 2: Core Implementation**
- Phase 2: Core modules
- Deliverable: All main classes/functions implemented

**Day 3: Integration & Testing**
- Phase 3: Integration, evaluation, testing
- Deliverable: End-to-end flow works, tests pass

**Day 4: Polish & Documentation**
- Phase 4: Application, documentation
- Deliverable: Demo app, README, ready for showcase

### Time Allocation Guidance

| Phase | Typical Allocation | Notes |
|-------|-------------------|-------|
| Phase 0 | 5-10% | Should be quick if reusing setup scripts |
| Phase 1 | 15-20% | Depends on data complexity |
| Phase 2 | 30-40% | Core development work |
| Phase 3 | 20-25% | Integration often reveals issues |
| Phase 4 | 15-20% | Don't underestimate documentation |

---

## 8. Integration with Standard DSM

This adaptation **extends** DSM, it doesn't replace it.

**Continue using from standard DSM:**
- Section 1.3: Core Philosophy (communication style, factual accuracy)
- Section 3: Communication & Working Style
- Section 6: Tools & Best Practices (where applicable)
- Appendix A: Environment Setup Details
- Appendix E: Quick Reference (file naming, commands)

**Replace with this adaptation:**
- Section 2: Core Workflow (use adapted phases)
- Section 4.1: Decision Log (use architectural template)
- Section 2.5: Communication phase (use documentation focus)

**Reference as needed:**
- PM Guidelines: For sprint planning structure
- Appendix C: Advanced practices (experiment tracking, testing)

---

## 9. Version History

**v1.0 (January 2026):**
- Initial release
- Created for DevFlow Analyzer project (JetBrains application)
- Adapted from DSM 1.1 for software engineering context
