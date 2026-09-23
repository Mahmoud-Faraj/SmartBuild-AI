# SmartBuild AI Data Card

## Dataset summary

The included dataset contains 4,320 deterministic synthetic hourly observations from January 1 through June
29, 2026. It represents a commercial building with weather, occupancy, four subsystem loads, total energy,
and known injected abnormal intervals.

## Generation process

The generator combines seasonal and daily temperature cycles, inverse temperature-humidity dependence,
weekday commercial occupancy, temperature-sensitive HVAC demand, occupied-hour lighting and plug loads,
stable critical loads, and random variation. Selected intervals receive elevated HVAC or plug-load demand.
The fixed random seed makes the release reproducible.

## Fields

| Field | Type | Description |
|---|---|---|
| `timestamp` | datetime | Hour of observation |
| `outdoor_temp_c` | float | Outdoor temperature |
| `humidity_pct` | float | Relative humidity |
| `occupancy` | integer | Estimated people present |
| `hvac_kwh` | float | HVAC energy |
| `lighting_kwh` | float | Lighting energy |
| `plug_load_kwh` | float | Plug-load energy |
| `critical_load_kwh` | float | Continuously operating load |
| `total_energy_kwh` | float | Prediction target |
| `injected_anomaly` | binary | Known synthetic abnormal interval |

## Appropriate use

The dataset supports software testing, demonstration, controlled anomaly evaluation, and educational use. It
does not represent a surveyed or metered facility.

## Limitations

Synthetic relationships are cleaner than real building behaviour. The data does not include missing sensors,
meter resets, tariffs, equipment states, holidays, renovations, or persistent concept drift. Results must not be
presented as evidence of performance on a real building.
