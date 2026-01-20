# Data Science for Residential Energy Systems

A domain knowledge repository bridging energy engineering fundamentals with data science and machine learning applications for building energy optimization.

**Focus:** Systematic documentation of German heating standards (DIN, VDI, GEG) with applied ML methodologies for residential heating system optimization.

---

## Motivation

The energy sector faces a critical skills gap. According to the [IEA World Energy Employment 2025 report](https://www.iea.org/news/energy-employment-has-surged-but-growing-skills-shortages-threatening-future-momentum), **60% of energy companies report labor shortages**, with particular demand for data analytics and ML expertise. Meanwhile, [academic reviews](https://www.cambridge.org/core/journals/environmental-data-science/article/machine-learning-for-smart-and-energyefficient-buildings/CF271F74CEE670ACFA6AA7AAB9798416) note that resources bridging ML research and building energy applications remain fragmented across disciplines.

This repository addresses that gap through:

- Comprehensive documentation of German heating standards for data science practitioners
- ML methodologies specifically adapted for energy time series analysis
- Production-grade MLOps patterns for IoT and sensor data systems
- Applied case studies demonstrating real-world optimization approaches

---

## Repository Structure

| Part | Topic | Scope |
|------|-------|-------|
| [Part I](01_Part_I_Domain_Fundamentals.md) | Domain Fundamentals | Thermodynamics, heating system types, German standards |
| [Part II](02_Part_II_Data_Science_ML.md) | Data Science & ML | Time series analysis, forecasting, anomaly detection, optimization |
| [Part III](03_Part_III_Production_MLOps.md) | Production & MLOps | Data pipelines, deployment patterns, monitoring |
| [Part IV](04_Part_IV_Technical_Stack.md) | Technical Stack | Python, SQL, GraphQL, API design |
| [Part V](05_Part_V_Interview_Scenarios.md) | Applied Scenarios | Case studies, system design, cross-functional collaboration |
| [References](06_References.md) | References | Academic sources, regulations, technical glossary |

---

## Target Audience

- **Data Scientists** transitioning into energy and building optimization domains
- **Energy Engineers** integrating ML and data-driven approaches into practice
- **Researchers** in building science, smart buildings, and energy efficiency
- **Graduate Students** in energy systems, building physics, or applied ML

---

## Content Overview

### Part I: Domain Fundamentals

Systematic review of residential heating system engineering:

- **Chapter 1:** Thermodynamic principles (heat transfer mechanisms, thermal mass, U-values)
- **Chapter 2:** Heating system typology (gas boilers, heat pumps, district heating, CHP)
- **Chapter 3:** Control parameters (Heizkennlinie, Vorlauf/Rücklauf temperature management)
- **Chapter 4:** Hydraulic balancing methodology (Hydraulischer Abgleich)
- **Chapter 5:** Sector coupling (PV + heat pump + storage integration)

### Part II: Data Science & ML

Applied machine learning methodologies for energy optimization:

- **Chapter 6:** Time series fundamentals for energy data
- **Chapter 7:** Heat demand forecasting and energy production prediction
- **Chapter 8:** Anomaly detection frameworks for heating systems
- **Chapter 9:** Control and optimization algorithms (MPC, reinforcement learning)
- **Chapter 10:** Supervised learning applications
- **Chapter 11:** Unsupervised learning (building clustering, load profile segmentation)

### Part III: Production & MLOps

Deployment and operational considerations for ML systems:

- **Chapter 12:** Data pipeline architecture for IoT/energy systems (MQTT, TimescaleDB)
- **Chapter 13:** Production-grade algorithm development
- **Chapter 14:** MLOps frameworks (MLflow, experiment tracking)
- **Chapter 15:** Deployment patterns (batch inference, real-time scoring, edge deployment)

### Part IV: Technical Stack

Implementation reference for energy data systems:

- **Chapter 16:** Python for energy data science (Pandas, NumPy, scikit-learn)
- **Chapter 17:** Data access patterns (SQL window functions, GraphQL, REST APIs)

### Part V: Applied Scenarios

Case study analysis and system design:

- **Chapter 18:** Case study walkthroughs (district heating, heat pump systems, building portfolios)
- **Chapter 19:** System design exercises
- **Chapter 20:** Cross-functional collaboration patterns

---

## Technical Glossary (German-English)

| German | English | Technical Context |
|--------|---------|-------------------|
| Heizkennlinie | Heating curve | Flow temperature as function of outdoor temperature |
| Vorlauftemperatur | Flow/supply temperature | Water temperature leaving the heat source |
| Rücklauftemperatur | Return temperature | Water temperature returning to heat source |
| Spreizung | Temperature spread | Difference between flow and return (Vorlauf - Rücklauf) |
| Brennwertnutzung | Condensing operation | Recovering latent heat from flue gas |
| Hydraulischer Abgleich | Hydraulic balancing | Optimizing flow rates across heating circuits |
| Nachtabsenkung | Night setback | Reducing heating during unoccupied hours |
| Wärmepumpe | Heat pump | Device using refrigeration cycle for heating |
| Fernwärme | District heating | Centralized heat distribution network |
| BHKW | CHP (Combined Heat & Power) | Simultaneous electricity and heat generation |
| Anschlussleistung | Connection capacity | Contracted power for district heating |
| Wärmegestehungskosten | Heat generation costs | Total cost per kWh of useful heat |

See [06_References.md](06_References.md) for complete glossary.

---

## Key Performance Indicators

| Metric | Typical Range | Significance |
|--------|---------------|--------------|
| Flow temperature at 4°C outdoor | 50-70°C | Lower values indicate higher efficiency |
| Return temperature for condensing | < 55°C | Required threshold for gas boiler condensing operation |
| Heat pump COP | 3-5 | Coefficient of performance; higher = more efficient |
| Optimization savings potential | 10-20% | Achievable through control optimization |
| Systems without night setback | ~70% | Represents immediate optimization opportunity |
| Heat-pump-ready buildings | ~17% | Without major renovation requirements |

---

## Regulatory Framework (Germany)

| Regulation | Scope |
|------------|-------|
| **GEG (Gebäudeenergiegesetz)** | Building energy efficiency standards |
| **GEG §60b** | Mandatory heating system inspection requirements |
| **GEG §60c** | Hydraulic balancing requirements |
| **DIN EN 12831** | Heat load calculation methodology |
| **VDI 2067** | Economic calculation for energy systems |
| **VDI 6030** | Radiator sizing and selection |

---

## Contributing

Contributions are welcome. This repository aims to serve as a professional reference for the energy transition.

**Contribution areas:**
- Technical corrections and clarifications
- Code examples and implementation notebooks
- Translation to other languages
- Anonymized case studies

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Author

**Alberto Diaz Durana**

- Energy engineering background (TU Berlin) with data science specialization
- Professional experience in heating system optimization and ML deployment
- [GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)

---

## License

This work is licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose

Under the following terms:
- **Attribution** — Give appropriate credit and indicate if changes were made
- **ShareAlike** — Distribute contributions under the same license

---

## Acknowledgments

- German heating standards documentation based on DIN, VDI, and GEG regulations
- Weather data patterns informed by DWD (Deutscher Wetterdienst) open data
- Load profiles referenced from BDEW standard load profiles

---

**Status:** Active development
**Last Updated:** January 2026
