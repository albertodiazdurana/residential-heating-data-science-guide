# Data Science for Residential Energy Systems

An open educational resource combining energy engineering domain knowledge with machine learning techniques for heating system optimization in residential buildings.

**Focus:** German heating standards (DIN, VDI, GEG) explained for data scientists, with production-ready ML patterns for energy applications.

---

## Why This Repository?

The energy sector faces a critical skills gap. According to the [IEA World Energy Employment 2025 report](https://www.iea.org/news/energy-employment-has-surged-but-growing-skills-shortages-threaten-future-momentum), **60% of energy companies report labor shortages**, with particular demand for data analytics and ML expertise. Meanwhile, [academic reviews](https://www.cambridge.org/core/journals/environmental-data-science/article/machine-learning-for-smart-and-energyefficient-buildings/CF271F74CEE670ACFA6AA7AAB9798416) note that resources bridging ML research and building energy applications remain fragmented across disciplines.

This guide addresses that gap by providing:

- German heating standards (DIN, VDI, GEG) explained for data scientists
- ML techniques specifically adapted for energy time series
- Production MLOps patterns for IoT/sensor data
- Real-world case study patterns

---

## Quick Start

| Part | Topic | Best For |
|------|-------|----------|
| [Part I](01_Part_I_Domain_Fundamentals.md) | Domain Fundamentals | Understanding heating systems, thermodynamics, German standards |
| [Part II](02_Part_II_Data_Science_ML.md) | Data Science & ML | Time series, forecasting, anomaly detection, optimization |
| [Part III](03_Part_III_Production_MLOps.md) | Production & MLOps | Data pipelines, deployment, monitoring for IoT |
| [Part IV](04_Part_IV_Technical_Stack.md) | Technical Stack | Python, SQL, GraphQL, APIs for energy data |
| [Part V](05_Part_V_Interview_Scenarios.md) | Applied Scenarios | Case studies, system design exercises |
| [References](06_References.md) | References | Academic sources, regulations, glossary |

---

## Who Is This For?

- **Data Scientists** entering energy/building optimization domains
- **Energy Engineers** wanting to leverage ML and data science
- **Students** in energy systems, building science, or applied ML programs
- **Researchers** working on smart buildings and energy efficiency

---

## Content Overview

### Part I: Domain Fundamentals

Build foundational knowledge of residential heating systems:

- **Chapter 1:** Thermodynamic principles (heat transfer, thermal mass, U-values)
- **Chapter 2:** Heating system types (gas boilers, heat pumps, district heating, CHP)
- **Chapter 3:** Key control variables (Heizkennlinie, Vorlauf/Rücklauf temperatures)
- **Chapter 4:** Hydraulic balancing (Hydraulischer Abgleich)
- **Chapter 5:** Sector coupling (PV + heat pump + storage integration)

### Part II: Data Science & ML

Apply machine learning to energy optimization:

- **Chapter 6:** Time series fundamentals for energy data
- **Chapter 7:** Forecasting heat demand and energy production
- **Chapter 8:** Anomaly detection in heating systems
- **Chapter 9:** Control and optimization algorithms (MPC, RL concepts)
- **Chapter 10:** Supervised learning applications
- **Chapter 11:** Unsupervised learning (building clustering, load profiling)

### Part III: Production & MLOps

Deploy and maintain ML systems for energy applications:

- **Chapter 12:** Data pipelines for IoT/energy systems (MQTT, TimescaleDB)
- **Chapter 13:** Production-ready algorithm development
- **Chapter 14:** MLOps fundamentals (MLflow, experiment tracking)
- **Chapter 15:** Deployment patterns (batch vs real-time, edge deployment)

### Part IV: Technical Stack

Master the tools of the trade:

- **Chapter 16:** Python for energy data science (Pandas, NumPy, scikit-learn)
- **Chapter 17:** Data access patterns (SQL window functions, GraphQL, REST APIs)

### Part V: Applied Scenarios

Practice with realistic case studies:

- **Chapter 18:** Case study walkthroughs (district heating, heat pumps, building portfolios)
- **Chapter 19:** System design exercises
- **Chapter 20:** Cross-functional collaboration patterns

---

## German-English Technical Glossary

| German | English | Context |
|--------|---------|---------|
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

## Key Metrics Reference

| Metric | Typical Value | Significance |
|--------|---------------|--------------|
| Flow temp at 4°C outdoor | 50-70°C | Lower is more efficient |
| Return temp for condensing | < 55°C | Required for gas boiler efficiency |
| Heat pump COP | 3-5 | Higher = more efficient |
| Typical optimization savings | 10-20% | Achievable through control optimization |
| Systems without night setback | ~70% | Common optimization opportunity |
| Heat-pump-ready buildings | ~17% | Without major renovation |

---

## Regulatory Context (Germany)

| Regulation | Key Requirements |
|------------|------------------|
| **GEG (Gebäudeenergiegesetz)** | Building energy efficiency standards |
| **GEG §60b** | Mandatory heating system inspection |
| **GEG §60c** | Hydraulic balancing requirements |
| **DIN EN 12831** | Heat load calculation standard |
| **VDI 2067** | Economic calculation for energy systems |
| **VDI 6030** | Radiator sizing and selection |

---

## Related Resources

### Methodology Framework

This repository includes a Data Science Methodology (DSM) framework in the `DSM/` folder:

- Project management templates
- Notebook development standards
- Documentation best practices

---

## Contributing

Contributions are welcome! This repository aims to be a community resource for the energy transition.

**Ways to contribute:**
- Fix errors or clarify explanations
- Add code examples or notebooks
- Translate content to other languages
- Share case studies (anonymized)

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Author

**Alberto Diaz Durana**

- Background in energy engineering (TU Berlin) and data science
- Experience with heating system optimization and ML deployment
- [GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)

---

## License

This work is licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — Give appropriate credit
- **ShareAlike** — Distribute contributions under the same license

---

## Acknowledgments

- German heating standards documentation based on DIN, VDI, and GEG regulations
- Weather data patterns informed by DWD (Deutscher Wetterdienst) open data
- Load profiles referenced from BDEW standard profiles

---

**Status:** Active development
**Last Updated:** January 2026
