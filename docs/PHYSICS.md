# Physics - Solar-Forecast (PhysSolar)

The physical clear-sky model is the forecasting PRIOR: expected PV output under cloudless skies.
ML learns the weather-driven residual on top of it.

## 1. Solar geometry  (`physics.cos_zenith`, numpy, no deps)
- Declination:  delta = 23.45 deg * sin( 360/365 * (doy - 81) )
- Local solar time ~ UTC_hour + longitude/15 ;  hour angle  h = 15 deg * (solar_time - 12)
- Solar zenith Z:  cos(Z) = sin(lat) sin(delta) + cos(lat) cos(delta) cos(h)   ( >0 => daytime )
- Air mass ~ 1 / cos(Z)  (Kasten-Young at low sun).

## 2. Clear-sky irradiance (GHI)  (`physics.clear_sky_ghi`)
- Preferred: **pvlib Ineichen-Perez** - `Location.get_clearsky(times, model="ineichen")` (Linke
  turbidity climatology; validated).
- Fallback (no pvlib): **Haurwitz** -  GHI_cs = 1098 * cos(Z) * exp(-0.059 / cos(Z))  for cos(Z)>0, else 0.

## 3. Plane-of-array (POA)  [Phase 2 refinement]
Transpose GHI/DNI/DHI onto the tilted panel plane (tilt, azimuth):
`pvlib.irradiance.get_total_irradiance(...)`. v0 uses GHI directly; POA is the next accuracy step.

## 4. PV power model  (`physics.pv_power`)
- P = A * eta * G * PR * [ 1 + gamma * (T_cell - 25) ]
  - A = panel area (m^2), eta = module efficiency, G = irradiance (W/m^2),
    PR = performance ratio, gamma = temperature coefficient (~ -0.004 /degC).
- Cell temperature (simple):  T_cell ~ T_ambient + 0.025 * G  (a NOCT-style rise).
  Better: `pvlib.temperature.sapm_cell` / `faiman`.
- The clear-sky prior is `physics.clear_sky_power(site, index, temp_c)`.

## pvlib functions (when installed)
- `pvlib.location.Location.get_solarposition(times)`
- `pvlib.location.Location.get_clearsky(times, model="ineichen")`
- `pvlib.irradiance.get_total_irradiance(...)`   (POA)
- `pvlib.temperature.sapm_cell(...)`             (cell temperature)

## Pipeline sketch
clear_sky_ghi(site, t)  ->  pv_power(ghi, temp, site)  ->  clear_sky_power  (the prior the ML corrects)
