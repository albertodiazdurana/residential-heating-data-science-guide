# Data Science for Residential Energy Systems - Interview Preparation

**Author:** Alberto Diaz Durana
**Target Position:** Senior Data Scientist / Energy Engineer - Green Fusion GmbH
**Domain:** AI-Powered Heating Optimization for German Residential Buildings
**Last Updated:** January 2026

---

## Project Overview

This repository contains comprehensive study materials and technical preparation for roles at Green Fusion GmbH, a Berlin-based cleantech startup specializing in AI-powered heating system optimization for the German housing sector (Wohnungswirtschaft).

### Target Roles

1. **Senior Data Scientist (m/f/d)** - Energy Tech Start Up
   - Full product lifecycle: feasibility studies to deployment
   - Production-ready ML algorithms for energy optimization
   - Time series forecasting, anomaly detection, control algorithms
   - Cross-functional collaboration with energy engineers

2. **Energy Engineer (f/m/d)** - NEW POSITION
   - Domain expert in cross-functional product teams
   - Bridge between energy engineering, data, and product teams
   - Analyze heating systems with data scientists
   - Translate physical system understanding into models, rules, and KPIs
   - Design meaningful analyses for customer platform

---

## Repository Structure

### Core Study Materials

| File | Content | Size |
|------|---------|------|
| **[00_TOC.md](00_TOC.md)** | Master table of contents for 20-chapter study guide | 5.2KB |
| **[08_Project_Handoff.md](08_Project_Handoff.md)** | Complete project context and setup instructions | 19KB |
| **[07_Job_Description_Mapping.md](07_Job_Description_Mapping.md)** | Maps job requirements to study guide sections | 18KB |

### Study Guide - Part I: Domain Fundamentals (22KB)

**[01_Part_I_Domain_Fundamentals.md](01_Part_I_Domain_Fundamentals.md)**

- Chapter 1: Thermodynamic Principles for Heating Systems
- Chapter 2: Heating System Types & Control Parameters
- Chapter 3: Key Control Variables (Heizkennlinie, Vorlauf/Rücklauf)
- Chapter 4: Hydraulic Balancing (Hydraulischer Abgleich)
- Chapter 5: Sector Coupling (Sektorkopplung)

### Study Guide - Part II: Data Science & ML (34KB)

**[02_Part_II_Data_Science_ML.md](02_Part_II_Data_Science_ML.md)**

- Chapter 6: Time Series Fundamentals for Energy Data
- Chapter 7: Forecasting Heat Demand & Energy Production
- Chapter 8: Anomaly Detection in Heating Systems
- Chapter 9: Control & Optimization Algorithms
- Chapter 10: Supervised Learning Applications
- Chapter 11: Unsupervised Learning Applications

### Study Guide - Part III: Production & MLOps (42KB)

**[03_Part_III_Production_MLOps.md](03_Part_III_Production_MLOps.md)**

- Chapter 12: Data Pipelines for IoT/Energy Systems
- Chapter 13: Production-Ready Algorithm Development
- Chapter 14: MLOps Fundamentals
- Chapter 15: Deployment Patterns

### Study Guide - Part IV: Technical Stack (39KB)

**[04_Part_IV_Technical_Stack.md](04_Part_IV_Technical_Stack.md)**

- Chapter 16: Python for Energy Data Science
- Chapter 17: Data Access Patterns (SQL, GraphQL, REST)

### Study Guide - Part V: Interview Scenarios (56KB)

**[05_Part_V_Interview_Scenarios.md](05_Part_V_Interview_Scenarios.md)**

- Chapter 18: Case Study Walkthroughs
- Chapter 19: System Design Questions
- Chapter 20: Behavioral & Cross-Functional Collaboration

### References & Regulations

**[06_References.md](06_References.md)** (7.8KB)

- German regulations (GEG, EnSimiMaV)
- Academic sources and textbooks
- Industry standards
- German-English technical glossary

### Technical Gaps & Bridge Documents

| File | Content | Size |
|------|---------|------|
| **[GreenFusion_Technical_Gaps_StudyGuide_TimeSeriesForecasting.md](GreenFusion_Technical_Gaps_StudyGuide_TimeSeriesForecasting.md)** | Advanced concepts: RL, MPC, Multi-objective optimization | 48KB |
| **[GreenFusion_Technical_Gaps_StudyGuide_Segmentation.md](GreenFusion_Technical_Gaps_StudyGuide_Segmentation.md)** | TravelTide segmentation → Building clustering translation | 22KB |

---

## Green Fusion GmbH - Company Context

### Company Profile

- **Founded:** 2021 by Paul Hock, Simon Wagenknecht, Nina Germanus
- **Location:** Berlin (Hohen Neuendorf), Germany
- **Funding:** €12M Series A (January 2025), led by HV Capital and XAnge
- **Market Position:** Market leader in heating monitoring and optimization per GdW
- **Scale:** 100+ housing company customers, 800,000+ apartments served

### Core Product

- **Energiespar-Pilot / Energiespar-Autopilot:** Cloud-based AI platform
- **GreenBox:** Hardware gateway for heating system connectivity
- **Average Results:** 16% heating cost/energy reduction
- **Technology:** Hardware-agnostic, supports all heating system types

### Heating System Types Supported

- Gas/oil boilers (condensing technology)
- District heating (Fernwärme)
- Heat pumps (all source types)
- CHP units (BHKW)
- Hybrid systems
- Sector coupling (PV + heat pump + storage)

---

## Key Technical Concepts

### Domain Metrics to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| Average Vorlauf at 4°C outdoor | 64.4°C | Too high; optimization potential |
| Systems without night setback | 70% | Quick win opportunity |
| Average savings from optimization | 16% | Green Fusion benchmark |
| Heat pump COP range | 3-5 | 3-5x electrical input |
| Condensing threshold (gas) | Return < 55°C | Required for Brennwertnutzung |
| Heat-pump-ready buildings | 17% | Without renovation |
| Legionella risk threshold | Circulation < 50°C | 10% of systems |

### Essential German Technical Terms

| German | English | Technical Context |
|--------|---------|-------------------|
| Heizkennlinie | Heating curve | Flow temp = f(outdoor temp) |
| Vorlauftemperatur | Flow temperature | Supply to heating circuits |
| Rücklauftemperatur | Return temperature | Back to heat source |
| Spreizung | Temperature spread | Vorlauf - Rücklauf |
| Anschlussleistung | Connection power | Fernwärme capacity charge |
| Brennwertnutzung | Condensing operation | Latent heat recovery |
| Hydraulischer Abgleich | Hydraulic balancing | Flow rate optimization |
| Nachtabsenkung | Night setback | Reduced heating at night |

### Regulatory Context

- **GEG §60b:** Mandatory heating inspection and optimization
- **GEG §60c:** Hydraulic balancing requirements
- **GEG §71a:** TÜV certification for optimization systems
- **DIN EN 12831:** Heat load calculation standard
- **EnSimiMaV:** Expired Sept 2024 (historical context)

---

## Technical Stack Requirements

### Core Technologies

**Programming & Data Science:**
- Python (Pandas, NumPy, SciPy, Scikit-learn, LightGBM, TensorFlow)
- SQL (window functions, time series queries)
- GraphQL (schema design, hierarchical data)
- REST APIs (versioning, pagination, error handling)

**Data Infrastructure:**
- Time-series databases: TimescaleDB, InfluxDB
- Message queues: MQTT for IoT devices
- Data pipelines: ETL patterns for sensor data
- Cloud-to-cloud integrations

**MLOps:**
- Experiment tracking: MLflow, Weights & Biases
- CI/CD pipelines for ML
- Model monitoring: data drift, performance degradation
- Version control: Git, collaborative coding

**Testing & Quality:**
- Unit testing: pytest
- Integration testing
- Property-based testing
- Code quality: type hints, docstrings, SOLID principles

---

## Use Cases: Claude Project Instructions

### Quick Reference Format

Store this in `.claude/CLAUDE.md` or Project Knowledge:

```markdown
# Project Context
This project supports preparation for data science/energy engineering roles at Green Fusion GmbH,
specializing in AI-powered heating optimization for German residential buildings.

## Key References
- Project Handoff: 08_Project_Handoff.md (complete context)
- Job Mapping: 07_Job_Description_Mapping.md
- Study Guide TOC: 00_TOC.md
- Domain Fundamentals: 01_Part_I_Domain_Fundamentals.md
- ML Techniques: 02_Part_II_Data_Science_ML.md
- Production Engineering: 03_Part_III_Production_MLOps.md
- Technical Stack: 04_Part_IV_Technical_Stack.md
- Interview Scenarios: 05_Part_V_Interview_Scenarios.md
- Regulations & References: 06_References.md
- Technical Gaps: GreenFusion_Technical_Gaps_StudyGuide_*.md

## Response Guidelines
- Use Celsius for all temperatures
- Retain German technical terms with English explanations
- Provide Python code examples when relevant
- Reference specific chapters when applicable
- Be rigorous and precise - senior role preparation

## Key Metrics to Remember
- Average Vorlauf at 4°C outdoor: 64.4°C (too high)
- Systems without night setback: 70%
- Average savings from optimization: 16%
- Heat pump COP range: 3-5
- Condensing threshold for gas: return temp < 55°C
```

### Conversation Starters

**Concept Review:**
```
"Explain [Heizkennlinie/COP/hydraulic balancing] as I would in an interview"
"What's the relationship between return temperature and condensing operation?"
```

**Technical Practice:**
```
"Write Python code to [detect anomalies in heating data / optimize heating curve / calculate COP]"
"How would I design a forecasting pipeline for heat demand?"
```

**Interview Prep:**
```
"Quiz me on [Part I domain concepts / ML techniques / system design]"
"How should I answer: 'Describe a time you worked with domain experts'?"
"What questions should I ask the interviewer about Green Fusion's tech stack?"
```

**Case Study Practice:**
```
"Walk me through the WSL Leipzig optimization as if I'm presenting to a customer"
"What would you look for first when diagnosing high consumption in a heat pump system?"
```

---

## Interview Preparation Checklist

### High Priority (Core Role)

- [ ] Heating curve optimization - Explain Heizkennlinie, slope/shift adjustments
- [ ] Time series forecasting - Heat demand prediction, weather integration
- [ ] Production ML pipelines - Scikit-learn, testing, deployment
- [ ] Case study analysis - Walk through WSL Leipzig or GWU Eckernförde

### Medium Priority (Differentiators)

- [ ] Control algorithms - MPC basics, RL for HVAC conceptually
- [ ] System design - Platform architecture for building portfolio
- [ ] Anomaly detection - Statistical + ML approaches, domain-specific

### Supporting (Demonstrate Breadth)

- [ ] MLOps practices - MLflow, monitoring, CI/CD
- [ ] Cross-functional collaboration - Engineer communication, requirements
- [ ] German regulatory context - GEG awareness

### Domain Knowledge

- [ ] Describe all heating system types (Part I, Chapter 2)
- [ ] Explain hydraulic balancing Verfahren A vs B (Part I, Chapter 4)
- [ ] Discuss sector coupling challenges (Part I, Chapter 5)
- [ ] Reference German regulations: GEG §60b, §60c

### Case Studies

- [ ] WSL Leipzig: district heating, 16.5% savings (Part V, 18.1)
- [ ] GWU Eckernförde: heat pump cascade, backup heater issue (Part V, 18.2)
- [ ] DIE EHRENFELDER: gas cascade, legacy digitalization (Part V, 18.3)

### Behavioral

- [ ] Collaboration with domain experts (Part V, 20.1)
- [ ] Translating customer requirements (Part V, 20.2)
- [ ] Communicating results to non-technical stakeholders (Part V, 20.3)
- [ ] STAR responses prepared (Part V, 20.4)

---

## Project History & Development

### Session 1: Initial Research & Study Guide Creation

**Accomplished:**
1. Researched Green Fusion GmbH via web search (funding, partnerships, product details)
2. Analyzed job description and mapped to data science competencies
3. Reviewed 4 detailed case studies
4. Reviewed 8 technical blog posts from Green Fusion website
5. Created comprehensive 20-chapter study guide (~22,000 words)
6. Created job description to study guide mapping
7. Prepared project handoff document

### Session 2: Energy Engineer Role Integration

**New Position Added:**
- Energy Engineer (f/m/d) job description analyzed
- Role comparison with Data Scientist position
- Hybrid role focus: domain expertise + data collaboration

### Quality Standards Applied

- All temperatures in Celsius
- German technical terms retained with English explanations
- Python code examples throughout
- Domain-specific numerical examples (actual Green Fusion metrics)
- Production-quality code patterns (type hints, docstrings, testing)
- Interview-focused structure with behavioral preparation

---

## Additional Resources

### Data Science Methodology (DSM)

This project follows the Data Science Methodology framework documented in the `DSM/` directory:

- **DSM_0_START_HERE_Complete_Guide.md:** System overview and setup
- **DSM_1.0_Data_Science_Collaboration_Methodology_v1.1.md:** 4-phase workflow
- **DSM_1.0_Methodology_Appendices.md:** Deep dives and examples
- **DSM_2.0_ProjectManagement_Guidelines_v2_v1.1.md:** Project planning framework
- **DSM_3_Methodology_Implementation_Guide_v1.1.md:** Setup instructions

### Related Projects

**TravelTide Customer Segmentation:**
- Demonstrates unsupervised learning (hierarchical clustering)
- 65-feature engineering pipeline
- Propensity scoring methodology
- Transferable to building portfolio segmentation
- Reference: [GreenFusion_Technical_Gaps_StudyGuide_Segmentation.md](GreenFusion_Technical_Gaps_StudyGuide_Segmentation.md)

**Favorita Demand Forecasting:**
- Demonstrates time series forecasting (LSTM, XGBoost)
- Temporal validation and feature engineering
- Production deployment experience
- Transferable to heat demand prediction
- Reference: [GreenFusion_Technical_Gaps_StudyGuide_TimeSeriesForecasting.md](GreenFusion_Technical_Gaps_StudyGuide_TimeSeriesForecasting.md)

---

## Contact & Author Information

**Alberto Diaz Durana**
- GitHub: [@albertodiazdurana](https://github.com/albertodiazdurana)
- LinkedIn: [albertodiazdurana](https://www.linkedin.com/in/albertodiazdurana/)

---

## License

This is a personal study and interview preparation repository. All Green Fusion information sourced from public materials (website, press releases, case studies).

---

**Document Status:** Complete and ready for interview preparation
**Total Study Material:** ~200KB of technical content across 20 chapters
**Preparation Level:** Senior Data Scientist / Energy Engineer position
