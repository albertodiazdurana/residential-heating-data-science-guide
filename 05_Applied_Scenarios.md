# Part V: Applied Scenarios

---

## Chapter 18: Case Study Walkthroughs

This chapter analyzes real-world case studies demonstrating how to translate business problems into data science solutions. These examples illustrate production optimization system development patterns.

### 18.1 District Heating Optimization: WSL Leipzig

**Context:**
The WSL Wohnen & Service Leipzig GmbH operates a 1950s residential building with 27 units (1,062 m²) served by a 30-year-old district heating system (90 kW Samson Trovis controller). The system lacked digital connectivity and operated without coordinated control.

**Business Problem:**
High energy consumption relative to building characteristics, no visibility into system performance, inability to optimize without on-site manual adjustments.

**Data Science Approach:**

**Phase 1: Digitalization and Data Collection**

The first step required establishing data infrastructure where none existed:

The following data sources were integrated via an IoT gateway, sampled at 15-minute intervals and transmitted through the Vodafone API to a cloud platform:

**Primary sensors:**
- Vorlauftemperatur (flow temperature): Anlegefuehler on supply pipe (°C)
- Ruecklauftemperatur (return temperature): Anlegefuehler on return pipe (°C)
- Outdoor temperature: External sensor (°C)
- Flow rate: Ultrasonic flow meter (L/min)

**Meter data:**
- Waermemengenzaehler (heat meter): Reading via M-Bus (kWh)

**External data:**
- Weather forecast: DWD API (temperature, wind, radiation)

**Phase 2: Baseline Analysis (4 weeks)**

Before optimization, establish baseline performance metrics:

```python
def analyze_baseline(df: pd.DataFrame) -> dict:
    """Analyze pre-optimization system performance."""
    
    # Energy consumption patterns
    daily_consumption = df['energy_kwh'].resample('D').sum()
    
    # Temperature analysis
    avg_vorlauf = df['vorlauf_temp'].mean()
    avg_ruecklauf = df['ruecklauf_temp'].mean()
    avg_spreizung = avg_vorlauf - avg_ruecklauf
    
    # Heating curve analysis - actual vs. optimal
    heating_curve_data = df.groupby(
        pd.cut(df['outdoor_temp'], bins=range(-10, 20, 2))
    )['vorlauf_temp'].mean()
    
    # Identify inefficiencies
    issues = []
    
    # Check for excessive flow temperatures
    if avg_vorlauf > 60:
        issues.append({
            'type': 'high_vorlauf',
            'current': avg_vorlauf,
            'target': 55,
            'potential_savings_pct': (avg_vorlauf - 55) * 1.5  # ~1.5% per °C
        })
    
    # Check for poor spreizung (indicates hydraulic issues)
    if avg_spreizung < 15:
        issues.append({
            'type': 'low_spreizung',
            'current': avg_spreizung,
            'target': 20,
            'potential_savings_pct': 5
        })
    
    # Check for missing night setback
    night_hours = df.between_time('22:00', '06:00')
    day_hours = df.between_time('08:00', '20:00')
    if night_hours['vorlauf_temp'].mean() >= day_hours['vorlauf_temp'].mean() - 1:
        issues.append({
            'type': 'no_night_setback',
            'potential_savings_pct': 8
        })
    
    return {
        'baseline_consumption_kwh_day': daily_consumption.mean(),
        'avg_vorlauf_temp': avg_vorlauf,
        'avg_spreizung': avg_spreizung,
        'heating_curve': heating_curve_data.to_dict(),
        'identified_issues': issues,
        'total_potential_savings_pct': sum(i['potential_savings_pct'] for i in issues)
    }
```

**Phase 3: Optimization Implementation**

Dynamic heating curve control was the primary intervention:

```python
class DynamicHeatingCurveController:
    """
    Adaptive heating curve controller for district heating.
    
    Adjusts flow temperature setpoint based on:
    - Outdoor temperature (primary)
    - Weather forecast (predictive)
    - Building thermal response (learned)
    - Time of day (scheduled setback)
    """
    
    def __init__(self, building_config: dict, model_params: dict):
        self.config = building_config
        self.params = model_params
        self.thermal_model = self._load_thermal_model()
    
    def calculate_setpoint(self, 
                           current_outdoor: float,
                           forecast_outdoor_6h: float,
                           current_hour: int,
                           current_indoor: float) -> float:
        """Calculate optimal Vorlauf temperature setpoint."""
        
        # Base heating curve
        base_setpoint = (
            self.params['base_temp'] + 
            self.params['slope'] * (20 - current_outdoor)
        )
        
        # Predictive adjustment: if warming trend, reduce now
        temp_trend = forecast_outdoor_6h - current_outdoor
        if temp_trend > 2:  # Warming by >2°C in next 6h
            predictive_reduction = min(3, temp_trend * 0.5)
            base_setpoint -= predictive_reduction
        
        # Night setback (22:00 - 06:00)
        if current_hour >= 22 or current_hour < 6:
            base_setpoint -= self.params['night_setback_reduction']
        
        # Comfort feedback: if indoor temp high, reduce further
        if current_indoor > self.config['comfort_max']:
            comfort_reduction = (current_indoor - self.config['comfort_max']) * 2
            base_setpoint -= comfort_reduction
        
        # Apply limits
        setpoint = np.clip(
            base_setpoint,
            self.params['min_vorlauf'],
            self.params['max_vorlauf']
        )
        
        return setpoint
```

**Phase 4: Results Measurement**

After 3 months of optimized operation:

```python
def measure_optimization_impact(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame
) -> dict:
    """Compare baseline to optimized performance."""
    
    # Weather-normalize consumption for fair comparison
    def consumption_per_hdd(df):
        daily = df.resample('D').agg({
            'energy_kwh': 'sum',
            'outdoor_temp': 'mean'
        })
        daily['hdd'] = (18 - daily['outdoor_temp']).clip(lower=0)
        return daily['energy_kwh'].sum() / daily['hdd'].sum()
    
    baseline_kwh_per_hdd = consumption_per_hdd(baseline_df)
    optimized_kwh_per_hdd = consumption_per_hdd(optimized_df)
    
    savings_pct = (1 - optimized_kwh_per_hdd / baseline_kwh_per_hdd) * 100
    
    return {
        'baseline_kwh_per_hdd': baseline_kwh_per_hdd,
        'optimized_kwh_per_hdd': optimized_kwh_per_hdd,
        'weather_normalized_savings_pct': savings_pct,
        'measurement_period_days': len(optimized_df.resample('D')),
        'confidence': 'high' if len(optimized_df) > 60*24*4 else 'preliminary'
    }
```

**Outcome:** 16.5% consumption reduction, achieved through dynamic Heizkennlinie adjustment and automated summer/winter mode switching. No comfort complaints from residents.

**Discussion Points:**
- Why weather normalization matters for comparing different time periods
- Trade-offs between aggressive optimization and comfort risk
- How the 4-week baseline period informs model parameters
- Regulatory context (EU SPARCS project, §60b GEG compliance)

---

### 18.2 Heat Pump Cascade: GWU Eckernförde

**Context:**
A 1972 building (16 units, 1,072 m²) underwent renovation in 2022-23, installing two Stiebel Eltron heat pumps with PV integration. Post-renovation, electricity consumption exceeded expectations.

**Business Problem:**
The renewable system was not delivering expected efficiency. Without detailed monitoring, the root cause was unknown. The housing cooperative needed to understand whether the issue was equipment, installation, or operational.

**Data Science Approach:**

**Phase 1: System Instrumentation**

Heat pump systems require comprehensive monitoring to diagnose efficiency issues:

The monitoring system covers five subsystems:

**Heat pumps (1 and 2, identical schema):**
- Electrical power (kW): compressor plus auxiliary power
- Thermal output (kW): heat delivered to buffer
- Source temperature: evaporator inlet (°C)
- Sink temperature: condenser outlet (°C)
- Compressor status: On / Off / Defrost
- COP instantaneous: calculated real-time COP

**Auxiliary systems:**
- Durchlauferhitzer (backup electric heater) power (kW)
- Circulation pump power (W)

**Storage:**
- Buffer tank temperatures: top, middle, and bottom (°C)
- DHW (domestic hot water) tank temperature (°C)

**PV system:**
- Generation (kW): current PV output
- Grid import (kW): power drawn from grid
- Grid export (kW): power fed to grid
- Self-consumption (kW): PV used on-site

**Phase 2: Root Cause Analysis**

Data analysis revealed the Durchlauferhitzer (instantaneous water heater) operating excessively:

```python
def analyze_heat_source_utilization(df: pd.DataFrame) -> dict:
    """Analyze which heat sources are serving load."""
    
    # Calculate energy contribution by source
    total_heat_energy = df['thermal_output_total_kwh'].sum()
    
    hp1_energy = df['hp1_thermal_kwh'].sum()
    hp2_energy = df['hp2_thermal_kwh'].sum()
    backup_energy = df['durchlauferhitzer_kwh'].sum()
    
    # Calculate effective COP including backup heater
    total_electrical = (
        df['hp1_electrical_kwh'].sum() +
        df['hp2_electrical_kwh'].sum() +
        df['durchlauferhitzer_kwh'].sum()  # COP = 1 for resistance heating
    )
    
    system_cop = total_heat_energy / total_electrical
    
    # Identify when backup runs unnecessarily
    backup_when_hp_available = df[
        (df['durchlauferhitzer_power_kw'] > 0.5) &
        (df['hp1_status'] == 'available') &
        (df['hp2_status'] == 'available') &
        (df['outdoor_temp'] > -5)  # HPs should handle this
    ]
    
    unnecessary_backup_hours = len(backup_when_hp_available) / 4  # 15-min data
    unnecessary_backup_kwh = backup_when_hp_available['durchlauferhitzer_kwh'].sum()
    
    # If HP had provided this heat at COP=3, savings would be:
    potential_savings_kwh = unnecessary_backup_kwh * (1 - 1/3)
    
    return {
        'hp_share_pct': (hp1_energy + hp2_energy) / total_heat_energy * 100,
        'backup_share_pct': backup_energy / total_heat_energy * 100,
        'system_cop': system_cop,
        'target_cop': 3.5,  # Expected for this installation
        'unnecessary_backup_hours': unnecessary_backup_hours,
        'unnecessary_backup_kwh': unnecessary_backup_kwh,
        'potential_savings_kwh': potential_savings_kwh,
        'root_cause': 'Durchlauferhitzer control threshold too aggressive'
    }
```

**Phase 3: Optimization Strategy**

The solution required coordinated control of the heat pump cascade:

```python
class HeatPumpCascadeOptimizer:
    """
    Optimize operation of heat pump cascade with backup heater.
    
    Objectives:
    1. Maximize heat pump utilization (high COP)
    2. Minimize backup heater runtime (COP = 1)
    3. Maintain comfort and DHW availability
    4. Maximize PV self-consumption
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.hp_capacity_kw = config['hp1_capacity'] + config['hp2_capacity']
        
    def determine_operating_mode(self, state: dict) -> dict:
        """Determine optimal operating mode for current conditions."""
        
        heat_demand = state['current_heat_demand_kw']
        pv_available = state['pv_generation_kw'] - state['base_load_kw']
        outdoor_temp = state['outdoor_temp']
        buffer_temp = state['buffer_temp_middle']
        dhw_temp = state['dhw_temp']
        
        commands = {
            'hp1': 'off',
            'hp2': 'off',
            'durchlauferhitzer': 'off',
            'target_buffer_temp': 45
        }
        
        # Priority 1: DHW legionella protection
        if dhw_temp < 55:
            commands['hp1'] = 'dhw_priority'
            commands['target_dhw_temp'] = 60
            return commands
        
        # Priority 2: Use PV excess for thermal storage
        if pv_available > 2.0 and buffer_temp < 55:
            commands['hp1'] = 'on'
            if pv_available > self.config['hp1_capacity'] + 1:
                commands['hp2'] = 'on'
            commands['target_buffer_temp'] = 55  # Store heat
            return commands
        
        # Priority 3: Meet heat demand with heat pumps
        if heat_demand > 0:
            # Estimate HP capacity at current conditions
            hp_cop = self._estimate_cop(outdoor_temp, commands['target_buffer_temp'])
            hp_capacity = self._capacity_at_conditions(outdoor_temp)
            
            if heat_demand <= hp_capacity * 0.6:
                commands['hp1'] = 'on'
            elif heat_demand <= hp_capacity:
                commands['hp1'] = 'on'
                commands['hp2'] = 'on'
            else:
                # Only use backup if HPs truly insufficient
                commands['hp1'] = 'on'
                commands['hp2'] = 'on'
                if buffer_temp < 35:  # Real emergency
                    commands['durchlauferhitzer'] = 'on'
        
        return commands
    
    def _estimate_cop(self, outdoor_temp: float, sink_temp: float) -> float:
        """Estimate COP at operating conditions."""
        # Simplified Carnot-based estimate with efficiency factor
        t_source = outdoor_temp + 273.15  # Kelvin
        t_sink = sink_temp + 273.15
        carnot_cop = t_sink / (t_sink - t_source)
        return carnot_cop * 0.45  # Typical 45% of Carnot
    
    def _capacity_at_conditions(self, outdoor_temp: float) -> float:
        """Estimate available HP capacity at outdoor temperature."""
        # Air-source HPs lose capacity in cold weather
        if outdoor_temp >= 7:
            factor = 1.0
        elif outdoor_temp >= 2:
            factor = 0.9
        elif outdoor_temp >= -7:
            factor = 0.75
        else:
            factor = 0.6
        return self.hp_capacity_kw * factor
```

**Phase 4: Results**

After implementing the cascade optimization:

- DHW electricity reduction: 20%
- Total system savings target: 15%
- System COP before optimization: 2.1
- System COP after optimization: 3.2
- Backup heater runtime reduction: 75%
- Key intervention: Raised backup heater activation threshold, prioritized heat pump operation, and implemented PV-synchronized charging

**Discussion Points:**
- How to diagnose efficiency issues in complex multi-source systems
- The importance of monitoring all energy flows, not just aggregate consumption
- Control hierarchy: comfort → efficiency → cost optimization
- PV self-consumption optimization as a time-shifting problem

---

### 18.3 Gas Boiler Cascade: DIE EHRENFELDER

**Context:**
A 1984/1987 building complex (>7,000 m² net floor area) with a gas boiler cascade serving multiple heating circuits and DHW storage. No existing digital interfaces; boilers running on factory settings.

**Business Problem:**
Excessive consumption with unknown root causes. No visibility into which boilers ran when, no coordinated cascade control, suspected inefficient operation.

**Data Science Approach:**

**Phase 1: Digitalization Challenge**

Legacy systems require creative instrumentation approaches:

The primary challenge was that the old gas boilers had no bus interfaces for digital communication. The solution used non-invasive retrofit instrumentation:

- **Temperature monitoring:** Anlegefuehler (clamp-on sensors) on pipes
- **Boiler status:** Current transformers on burner power supply
- **Gas consumption:** Pulse output from gas meter (retrofit)
- **Integration:** IoT gateway aggregating all signals

The installation added 24 data points in approximately 6 hours with minimal invasiveness and no interruption to heating service.

**Phase 2: Baseline Analysis Findings**

The baseline analysis identified four key issues:

**1. Simultaneous boiler operation.** All boilers were running simultaneously, even in summer. Burner status showed 3 out of 3 boilers active during DHW-only periods. Without cascade sequencing, each boiler responded to its own thermostat independently, causing excessive standby losses and poor part-load efficiency.

**2. Excessive DHW storage temperature.** The Speicher (DHW storage) was operating at 70°C instead of the target 60°C. Continuous temperature logging confirmed this was due to conservative factory settings, resulting in approximately 10% excess storage losses.

**3. Poor Spreizung (temperature spread).** The temperature difference across heating circuits was only 8-10K, as evidenced by Vorlauf/Ruecklauf logging. This was likely caused by excessive flow rates and possible hydraulic imbalance, reducing condensing operation and increasing pump energy consumption.

**4. No night setback (Nachtabsenkung).** The system operated at constant output 24/7 with no pattern change during night hours. Night setback had never been configured, leading to an estimated 8-10% excess consumption.

**Phase 3: Optimization Implementation**

```python
class GasBoilerCascadeController:
    """
    Intelligent control for gas boiler cascade.
    
    Optimizes:
    - Boiler sequencing (lead/lag rotation)
    - Condensing operation (maximize return temp < 55°C)
    - Part-load efficiency (avoid short cycling)
    - DHW scheduling (minimize storage losses)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.n_boilers = config['n_boilers']
        self.boiler_capacity_kw = config['boiler_capacity_kw']  # Per boiler
        self.min_modulation = config.get('min_modulation', 0.3)
        self.lead_boiler_index = 0
        self.runtime_hours = [0] * self.n_boilers
        
    def calculate_required_capacity(self, 
                                     heat_demand_kw: float,
                                     dhw_demand_kw: float) -> float:
        """Calculate total capacity requirement."""
        return heat_demand_kw + dhw_demand_kw
    
    def determine_active_boilers(self, required_kw: float) -> list:
        """
        Determine which boilers should run and at what output.
        
        Strategy:
        1. Use minimum boilers to meet demand
        2. Keep each boiler above minimum modulation
        3. Rotate lead boiler for even wear
        """
        if required_kw <= 0:
            return []
        
        # Calculate boilers needed
        capacity_per_boiler = self.boiler_capacity_kw * self.min_modulation
        min_boilers = int(np.ceil(required_kw / self.boiler_capacity_kw))
        min_boilers = max(1, min(min_boilers, self.n_boilers))
        
        # Check if min_boilers can meet demand above min modulation
        while min_boilers < self.n_boilers:
            per_boiler_load = required_kw / min_boilers
            if per_boiler_load >= capacity_per_boiler:
                break
            min_boilers += 1
        
        # Select boilers starting from lead
        active = []
        for i in range(min_boilers):
            boiler_idx = (self.lead_boiler_index + i) % self.n_boilers
            load_fraction = min(1.0, required_kw / min_boilers / self.boiler_capacity_kw)
            active.append({
                'boiler_index': boiler_idx,
                'load_fraction': load_fraction,
                'output_kw': load_fraction * self.boiler_capacity_kw
            })
        
        return active
    
    def optimize_for_condensing(self, 
                                 active_boilers: list,
                                 current_ruecklauf: float) -> list:
        """
        Adjust operation to maximize condensing (Brennwertnutzung).
        
        Condensing occurs when return temp < ~55°C for natural gas.
        """
        if current_ruecklauf > 55:
            # Too hot for condensing - reduce flow temp target
            for boiler in active_boilers:
                boiler['vorlauf_setpoint_reduction'] = min(5, current_ruecklauf - 50)
        else:
            for boiler in active_boilers:
                boiler['vorlauf_setpoint_reduction'] = 0
        
        return active_boilers
    
    def rotate_lead_boiler(self):
        """Rotate lead boiler for even runtime distribution."""
        self.lead_boiler_index = (self.lead_boiler_index + 1) % self.n_boilers
    
    def anti_cycling_check(self, 
                           boiler_index: int,
                           last_stop_time: datetime,
                           current_time: datetime) -> bool:
        """
        Prevent short cycling that damages boilers.
        
        Enforce minimum off-time of 5 minutes.
        """
        min_off_time = timedelta(minutes=5)
        if current_time - last_stop_time < min_off_time:
            return False  # Don't allow start
        return True
```

**Phase 4: Results and Expansion**

The optimization delivered substantial savings during the DHW-only period (exact percentage not published), with 15% expected savings over the full heating season. Key improvements included:

- Cascade sequencing preventing simultaneous boiler operation
- Reduced DHW storage temperature to 60°C
- Implemented Nachtabsenkung (night setback)
- Optimized Heizkennlinie (heating curve) for actual building thermal response
- Reduced unnecessary boiler starts through anti-cycling logic

The approach is being rolled out to 23 additional buildings, covering gas, pellet, and hybrid heat pump system types.

**Discussion Points:**
- Strategies for digitalizing legacy systems without bus interfaces
- Cascade control algorithms and boiler sequencing
- Why condensing operation matters (Brennwertnutzung) and how return temperature affects it
- Scaling optimization across heterogeneous building portfolios

---

## Chapter 19: System Design Questions

This chapter prepares you for system design situations with architecture questions relevant to energy management platforms.

### 19.1 Design: Energy Management Platform for 3,000 Buildings

**Prompt:** "Design a cloud platform to monitor and optimize heating systems across 3,000 multi-family buildings. The system should ingest sensor data, run optimization algorithms, and push control setpoints to building controllers."

**Clarifying Questions to Ask:**
- What's the sensor data volume? (Assume: 20 sensors/building, 15-min intervals = 2.88M readings/day)
- Latency requirements for control? (Assume: 15-minute control cycles acceptable)
- Geographic distribution? (Assume: primarily Germany, expanding EU)
- Failure tolerance requirements? (Assume: no single point of failure for data, graceful degradation for control)

**High-Level Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EDGE LAYER                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐           │
│  │ Gateway  │  │ Gateway  │  │ Gateway  │  ...  │ Gateway  │           │
│  │ Building1│  │ Building2│  │ Building3│       │ Building │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘           │
│       │             │             │                   │                 │
│       └─────────────┴──────┬──────┴───────────────────┘                 │
│                            │ MQTT/HTTPS                                 │
└────────────────────────────┼────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                     INGESTION LAYER                                      │
│                            ▼                                             │
│  ┌─────────────────────────────────────────┐                            │
│  │        MQTT Broker Cluster              │                            │
│  │        (EMQX / AWS IoT Core)            │                            │
│  └───────────────────┬─────────────────────┘                            │
│                      │                                                   │
│  ┌───────────────────▼─────────────────────┐                            │
│  │     Stream Processing (Kafka/Kinesis)    │                            │
│  │     - Validation, Enrichment             │                            │
│  │     - Real-time anomaly flagging         │                            │
│  └───────────────────┬─────────────────────┘                            │
└──────────────────────┼──────────────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────────┐
│                STORAGE LAYER                                             │
│                      ▼                                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────┐             │
│  │   Time-Series Database  │    │   Relational Database   │             │
│  │   (TimescaleDB/InfluxDB)│    │   (PostgreSQL)          │             │
│  │   - Sensor readings     │    │   - Building metadata   │             │
│  │   - 90 days hot         │    │   - User accounts       │             │
│  │   - 2 years warm (S3)   │    │   - Optimization logs   │             │
│  └─────────────────────────┘    └─────────────────────────┘             │
│                                                                          │
│  ┌─────────────────────────┐    ┌─────────────────────────┐             │
│  │   Object Storage (S3)   │    │   Model Registry        │             │
│  │   - Historical archives │    │   (MLflow)              │             │
│  │   - ML training data    │    │   - Versioned models    │             │
│  └─────────────────────────┘    └─────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────────┐
│             PROCESSING LAYER                                             │
│                      ▼                                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────┐             │
│  │  Batch Processing       │    │  Real-time Processing   │             │
│  │  (Airflow + Spark)      │    │  (Flink/Lambda)         │             │
│  │  - Daily aggregations   │    │  - Anomaly detection    │             │
│  │  - Model retraining     │    │  - Alert generation     │             │
│  │  - Report generation    │    │  - Live dashboards      │             │
│  └─────────────────────────┘    └─────────────────────────┘             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │              Optimization Engine                         │            │
│  │  - Building-specific models                              │            │
│  │  - Heating curve optimization                            │            │
│  │  - Setpoint calculation                                  │            │
│  │  - Scheduled execution (hourly)                          │            │
│  └─────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────────┐
│              APPLICATION LAYER                                           │
│                      ▼                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ API Gateway  │  │ Web Dashboard│  │ Mobile App   │                   │
│  │ (REST/GraphQL)│ │ (React)      │  │ (React Native)│                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │           Control Command Service                     │               │
│  │  - Setpoint distribution to buildings                 │               │
│  │  - Command acknowledgment tracking                    │               │
│  │  - Rollback capability                                │               │
│  └──────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

**Data Partitioning:**
Time-series data is partitioned by building_id (hash partition) and time (range partition, monthly). This enables efficient queries for single-building analysis and time-range aggregations across the portfolio.

**Retention policy:**

| Resolution | Retention |
|-----------|-----------|
| Raw 15-minute | 90 days |
| Hourly aggregates | 2 years |
| Daily aggregates | 10 years |

**Scaling Considerations:**
**Data volume:** With 20 sensors per building, 3,000 buildings, 96 readings per day (15-minute intervals), and approximately 100 bytes per reading, the system generates roughly 0.58 GB/day or 210 GB/year.

**Compute:** Running 3,000 optimization jobs per hour at an average of 2 seconds each requires approximately 1.7 parallel workers. Provisioning 10 workers provides headroom for traffic spikes.

**Fault Tolerance:**
**Edge offline:** When an edge gateway loses connectivity, it caches commands and executes the last-known-good setpoints for up to 24 hours. On reconnection, buffered data is automatically backfilled.

**Cloud partial failure:** The database layer uses multi-AZ PostgreSQL with read replicas. TimescaleDB provides time-series replication. Processing runs on Kubernetes with auto-scaling and automatic pod restart.

**Optimization failure:** If the optimization engine fails, the system maintains current setpoints and alerts operations. If the model becomes unavailable entirely, rule-based defaults serve as the fallback.

---

### 19.2 Design: Real-Time Anomaly Detection Pipeline

**Prompt:** "Design a system that detects anomalies in heating system sensor data in real-time and alerts operators within 5 minutes of occurrence."

**Architecture:**

```
Sensor Data → Kafka → Flink Processing → Alert Service → Notification
                           │
                           ▼
                    ┌──────────────┐
                    │ Anomaly      │
                    │ Detection    │
                    │ Models       │
                    │ (per building)│
                    └──────────────┘
```

**Flink Processing Logic:**

```python
# Conceptual Flink job (Python/PyFlink representation)

class AnomalyDetectionJob:
    """
    Real-time anomaly detection with Flink.
    
    Processing steps:
    1. Parse and validate incoming readings
    2. Enrich with building context
    3. Apply statistical bounds checks
    4. Apply ML model for complex anomalies
    5. Emit alerts for detected anomalies
    """
    
    def __init__(self):
        self.statistical_bounds = self._load_bounds()
        self.ml_models = {}  # Lazy-loaded per building
    
    def process_reading(self, reading: dict) -> Optional[Alert]:
        building_id = reading['building_id']
        sensor_id = reading['sensor_id']
        value = reading['value']
        timestamp = reading['timestamp']
        
        # Level 1: Hard bounds (physics-based)
        if not self._check_physical_bounds(sensor_id, value):
            return Alert(
                building_id=building_id,
                severity='critical',
                type='physical_violation',
                message=f'{sensor_id} value {value} outside physical bounds',
                timestamp=timestamp
            )
        
        # Level 2: Statistical bounds (learned from history)
        stats = self.statistical_bounds.get((building_id, sensor_id))
        if stats and not self._check_statistical_bounds(value, stats):
            return Alert(
                building_id=building_id,
                severity='warning',
                type='statistical_anomaly',
                message=f'{sensor_id} value {value} deviates from normal',
                timestamp=timestamp
            )
        
        # Level 3: ML-based (contextual anomalies)
        if building_id not in self.ml_models:
            self.ml_models[building_id] = self._load_model(building_id)
        
        model = self.ml_models[building_id]
        if model and model.is_anomaly(reading):
            return Alert(
                building_id=building_id,
                severity='warning',
                type='ml_anomaly',
                message=f'Contextual anomaly detected',
                timestamp=timestamp,
                details=model.explain(reading)
            )
        
        return None
    
    def _check_physical_bounds(self, sensor_id: str, value: float) -> bool:
        """Check against physically possible values."""
        bounds = {
            'vorlauf_temp': (20, 95),
            'ruecklauf_temp': (15, 80),
            'outdoor_temp': (-40, 50),
            'flow_rate_lpm': (0, 500),
        }
        sensor_type = sensor_id.split('_')[0] + '_' + sensor_id.split('_')[1]
        if sensor_type in bounds:
            return bounds[sensor_type][0] <= value <= bounds[sensor_type][1]
        return True
```

**Alert Aggregation:**

```python
class AlertAggregator:
    """
    Aggregate related alerts to prevent alert fatigue.
    
    - Group alerts by building and type within time window
    - Escalate severity if multiple related alerts
    - Deduplicate repeated alerts
    """
    
    def __init__(self, window_minutes: int = 15):
        self.window = timedelta(minutes=window_minutes)
        self.active_alerts = {}  # (building_id, type) -> list of alerts
    
    def add_alert(self, alert: Alert) -> Optional[AggregatedAlert]:
        key = (alert.building_id, alert.type)
        
        if key not in self.active_alerts:
            self.active_alerts[key] = []
        
        # Clean old alerts outside window
        cutoff = alert.timestamp - self.window
        self.active_alerts[key] = [
            a for a in self.active_alerts[key] 
            if a.timestamp > cutoff
        ]
        
        self.active_alerts[key].append(alert)
        
        # Emit aggregated alert if threshold reached
        if len(self.active_alerts[key]) >= 3:
            return AggregatedAlert(
                building_id=alert.building_id,
                alert_count=len(self.active_alerts[key]),
                severity='critical' if alert.severity == 'warning' else alert.severity,
                type=alert.type,
                first_occurrence=self.active_alerts[key][0].timestamp,
                latest_occurrence=alert.timestamp
            )
        
        # Emit single alert immediately for critical
        if alert.severity == 'critical':
            return AggregatedAlert.from_single(alert)
        
        return None  # Buffer warning-level alerts
```

---

### 19.3 Design: Multi-Tenant Data Architecture

**Prompt:** "Design the data architecture to support 100+ housing companies, each managing 10-1000 buildings, with strict data isolation requirements."

**Tenant Isolation Strategy:**

The chosen strategy is shared database with tenant column isolation. This approach is cost-effective for a large number of small-to-medium tenants, simplifies operations compared to database-per-tenant, and enforces isolation through row-level security.

**Implementation details:**
- Every table includes a `tenant_id` column
- Foreign keys use compound keys that include `tenant_id`
- All query indexes include `tenant_id` as a prefix
- Tenant-scoped views provide the interface for the application layer

**PostgreSQL Row-Level Security:**

```sql
-- Enable RLS on sensor_readings table
ALTER TABLE sensor_readings ENABLE ROW LEVEL SECURITY;

-- Create policy for tenant isolation
CREATE POLICY tenant_isolation ON sensor_readings
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Application sets tenant context on each connection
-- SET app.current_tenant = 'tenant-uuid-here';

-- Create tenant-scoped function for queries
CREATE OR REPLACE FUNCTION get_building_consumption(
    p_building_id TEXT,
    p_start_date TIMESTAMP,
    p_end_date TIMESTAMP
) RETURNS TABLE (
    date DATE,
    consumption_kwh NUMERIC
) AS $$
BEGIN
    -- tenant_id automatically filtered by RLS
    RETURN QUERY
    SELECT 
        DATE(timestamp) as date,
        SUM(energy_kwh) as consumption_kwh
    FROM sensor_readings
    WHERE building_id = p_building_id
      AND timestamp BETWEEN p_start_date AND p_end_date
    GROUP BY DATE(timestamp)
    ORDER BY date;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

**Application-Level Enforcement:**

```python
from contextvars import ContextVar
from functools import wraps

# Thread-safe tenant context
current_tenant: ContextVar[str] = ContextVar('current_tenant')

def tenant_required(func):
    """Decorator ensuring tenant context is set."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tenant_id = current_tenant.get(None)
        if not tenant_id:
            raise PermissionError("No tenant context set")
        return await func(*args, **kwargs)
    return wrapper

class TenantAwareRepository:
    """Repository with automatic tenant scoping."""
    
    def __init__(self, db_session):
        self.session = db_session
    
    async def _set_tenant_context(self):
        tenant_id = current_tenant.get()
        await self.session.execute(
            f"SET app.current_tenant = '{tenant_id}'"
        )
    
    @tenant_required
    async def get_buildings(self) -> List[Building]:
        await self._set_tenant_context()
        result = await self.session.execute(
            "SELECT * FROM buildings"  # RLS filters automatically
        )
        return [Building(**row) for row in result]
```

---

## Chapter 20: Behavioral & Cross-Functional Collaboration

This chapter prepares you for behavioral questions and discussions about working in cross-functional teams.

### 20.1 Working with Energy Engineers

**Scenario:** "Describe how you would collaborate with energy engineers who have deep domain expertise but limited data science background."

**Effective Collaboration Framework:**

**Respect domain expertise.** When an energy engineer states that "return temperature above 55°C prevents condensing," incorporate this as a hard constraint in the optimization, not a soft penalty. The anti-pattern is treating domain rules as suggestions that the model can override.

**Translate bidirectionally.** Communication must flow in both directions:
- *Data science to engineer:* Instead of saying "The model has 0.85 R-squared with RMSE of 3.2 kWh," say "The model predicts consumption within plus/minus 3 kWh 85% of the time, about as accurate as reading the meter with one decimal place."
- *Engineer to data science:* When they say "The heating curve is too steep," understand this as: the slope parameter in Vorlauf = f(outdoor) is too high, causing excessive flow temperatures in mild weather.

**Joint validation.** Review model outputs together before deployment. Key questions to ask the engineer:
- Does this recommendation make physical sense?
- Have you seen buildings behave this way?
- What could go wrong if we implement this?

**Feedback loops.** Establish a weekly review of optimization outcomes. Share energy savings, comfort complaints, and equipment alerts. The engineer's role is to explain unexpected patterns and suggest new features based on domain insight.

**Communication Example:**

```python
# Presenting anomaly detection results to energy engineers

def present_anomaly_findings(anomalies_df: pd.DataFrame) -> str:
    """Generate engineer-friendly anomaly report."""
    
    report = []
    report.append("## Heating System Anomalies Detected This Week\n")
    
    # Group by type for clarity
    for anomaly_type, group in anomalies_df.groupby('type'):
        report.append(f"### {anomaly_type.replace('_', ' ').title()}\n")
        
        if anomaly_type == 'high_return_temp':
            report.append(
                "Return temperatures exceeded 55°C, preventing condensing operation. "
                "This typically indicates:\n"
                "- Heating curve set too high for current weather\n"
                "- Possible hydraulic imbalance causing short-circuiting\n"
                "- Thermostatic valves fully open in some apartments\n"
            )
        
        # Show specific buildings affected
        report.append(f"Affected buildings: {len(group)}\n")
        for _, row in group.head(5).iterrows():
            report.append(
                f"- {row['building_address']}: "
                f"Rücklauf reached {row['max_value']:.1f}°C "
                f"on {row['timestamp'].strftime('%d.%m.%Y')}\n"
            )
        
        report.append("\n")
    
    return ''.join(report)
```

### 20.2 Translating Customer Requirements

**Scenario:** "A housing company says 'We want to save 20% on heating costs.' How do you translate this into a technical project?"

**Requirements Engineering Process:**

**Step 1: Clarify and quantify.** Before defining a technical approach, ask the customer:
- What is the baseline? (Last year's consumption? Average of 3 years?)
- Is the 20% target absolute or weather-normalized?
- What is the timeline? (This heating season? Over 2 years?)
- Are there constraints? (No comfort reduction? No capital investment?)
- How will success be measured? (Meter readings? Billing data?)

**Step 2: Define measurable success criteria.**
- Primary metric: weather-normalized energy consumption (kWh/HDD)
- Target: 20% reduction vs. baseline
- Baseline period: 2023-24 heating season
- Measurement period: 2024-25 heating season
- Comfort constraint: indoor temperature >= 20°C during occupied hours
- Measurement method: heat meter readings, monthly granularity

**Step 3: Identify technical interventions.**

*Phase 1 (quick wins):*
- Optimize Heizkennlinie (heating curve), expected 5-10% savings
- Implement Nachtabsenkung (night setback), expected 3-5% savings
- Reduce DHW storage temperature to 60°C, expected 2-3% savings

*Phase 2 (advanced):*
- Dynamic weather-predictive control, expected 3-5% additional savings
- Hydraulic balancing where data indicates need, expected 2-5% savings

Total expected range: 13-23%. Confidence: medium, depends on baseline system state.

**Step 4: Define project milestones.**

| Week | Deliverable |
|------|-------------|
| 0 | Edge gateway installation, data collection starts |
| 4 | Baseline analysis complete, optimization plan |
| 6 | Phase 1 optimizations deployed |
| 12 | First monthly savings report |
| 24 | Mid-season review, Phase 2 if needed |
| 40 | Full heating season results |

### 20.3 Communicating ML Results

**Scenario:** "How would you present model performance to a non-technical property manager?"

**Effective Communication Strategies:**

```python
def create_executive_summary(model_results: dict) -> str:
    """
    Create non-technical summary of model performance.
    """
    
    # Avoid: "RMSE improved from 4.2 to 3.1 kWh"
    # Better: Concrete business impact
    
    summary = f"""
## Heating Optimization Results: {model_results['building_name']}

### What We Achieved

Your heating system is now running more efficiently. Here's what that means:

**Energy Savings**
This month, your building used {model_results['savings_kwh']:,.0f} kWh less energy 
compared to what we'd expect for this weather. That's a 
{model_results['savings_pct']:.0f}% reduction.

**Cost Impact**
Based on your current energy rates, this translates to approximately 
€{model_results['savings_eur']:,.0f} saved this month.

**Comfort**
We received {model_results['comfort_complaints']} comfort complaints this period, 
compared to {model_results['baseline_complaints']} in the same period last year.

### How It Works

We continuously analyze your heating data and weather forecasts to determine 
the most efficient settings. The system automatically adjusts:

- **Flow temperatures**: Lowered by an average of {model_results['avg_temp_reduction']:.1f}°C 
  without affecting comfort
- **Night setback**: Heat is reduced by {model_results['night_setback']:.0f}°C from 
  10 PM to 6 AM when residents are typically sleeping
- **Weather anticipation**: When warming is forecast, we reduce heating earlier 
  to avoid wasting energy

### What's Next

Based on our analysis, we see additional opportunity to:
1. {model_results['next_recommendation']}

Would you like to discuss implementing this in our next review?
"""
    
    return summary
```

**Visualization Best Practices:**

```python
import matplotlib.pyplot as plt

def create_savings_visualization(monthly_data: pd.DataFrame,
                                  output_path: str):
    """
    Create simple, clear visualization for non-technical audience.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart: baseline vs. actual consumption
    months = monthly_data['month']
    x = range(len(months))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], 
                   monthly_data['expected_kwh'], 
                   width, 
                   label='Expected (without optimization)',
                   color='#ff7f7f')
    
    bars2 = ax.bar([i + width/2 for i in x], 
                   monthly_data['actual_kwh'], 
                   width, 
                   label='Actual (with optimization)',
                   color='#7fbf7f')
    
    # Add savings annotations
    for i, (exp, act) in enumerate(zip(monthly_data['expected_kwh'], 
                                        monthly_data['actual_kwh'])):
        savings_pct = (1 - act/exp) * 100
        ax.annotate(f'-{savings_pct:.0f}%', 
                   xy=(i + width/2, act),
                   ha='center', va='bottom',
                   fontweight='bold', color='green')
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title('Monthly Energy Savings from Optimization', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.legend()
    
    # Add total savings callout
    total_saved = monthly_data['expected_kwh'].sum() - monthly_data['actual_kwh'].sum()
    total_pct = (1 - monthly_data['actual_kwh'].sum() / 
                    monthly_data['expected_kwh'].sum()) * 100
    
    ax.text(0.02, 0.98, 
           f'Total Saved: {total_saved:,.0f} kWh ({total_pct:.0f}%)',
           transform=ax.transAxes,
           fontsize=12, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### 20.4 Sample Behavioral Questions and Responses

**Question:** "Tell me about a time when your model's recommendations were rejected by domain experts."

**STAR Response Framework:**

**Situation:** At a previous role, I developed an optimization model for HVAC scheduling that recommended running cooling systems at night to pre-cool buildings. The facilities team rejected this, saying it would increase energy costs.

**Task:** I needed to either validate my model's recommendations or understand what factor I had missed that the domain experts knew intuitively.

**Action:** I scheduled a working session with the lead facilities engineer. Instead of defending my model, I asked him to walk me through how he would approach the problem. I learned that electricity rates had a demand charge component I had not modeled; running at night would shift energy but create a new demand peak that increased the monthly bill. I updated the model to include demand charges, which completely changed the optimal schedule.

**Result:** The revised model was accepted and implemented. Energy costs decreased by 12% over three months. More importantly, I established a collaborative relationship with the facilities team; they now come to me with optimization ideas because they trust the process. Key lesson: domain experts often have valid information encoded as intuition. The data scientist's job is to extract and formalize that knowledge.

**Question:** "How do you handle disagreements about model approaches with other data scientists?"

**Approach:** I focus on defining clear evaluation criteria before debating approaches. For example, if a colleague prefers gradient boosting while I think a neural network would work better, we first agree on: (1) the exact metric, such as RMSE on time-series validation; (2) the validation methodology, such as walk-forward with a 24-hour gap; and (3) computational constraints, such as inference under 5 seconds for production. Then we implement both and compare. Data resolves most disagreements.

**When data is ambiguous:** If results are similar, we consider interpretability requirements (important in the regulated energy sector), maintainability (who else needs to understand this code?), and robustness to distribution shift. I am willing to defer to a colleague's preference if the objective metrics are equivalent; team cohesion matters more than being right.

---

*End of Part V*
