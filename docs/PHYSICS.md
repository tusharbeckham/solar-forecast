# Physics - Solar-Forecast (PhysSolar)

The physical clear-sky model is the forecasting PRIOR: expected PV output under cloudless skies.
ML learns the weather-driven residual on top of it.

## 1. Solar geometry  (`physics.cos_zenith`, numpy, no deps)
- Declination:  delta = 23.45 deg * sin( 360/365 * (doy - 81) )
- Local solar time ~ UTC_hour + longitude/15 ;  hour angle  h = 15 deg * (solar_time - 12)
- Solar zenith Z:  cos(Z) = sin(lat) sin(delta) + cos(lat) cos(delta) cos(h)   ( >0 => daytime )
- Air mass ~ 1 / cos(Z)  (Kasten-Young at low sun).
- Solar **azimuth** and the panel **angle of incidence (AOI)** are computed for POA transposition (`physics.cos_aoi`).

## 2. Clear-sky irradiance (GHI)  (`physics.clear_sky_ghi`)
- Preferred: **pvlib Ineichen-Perez** - `Location.get_clearsky(times, model="ineichen")` (Linke
  turbidity climatology; validated).
- Fallback (no pvlib): **Haurwitz** -  GHI_cs = 1098 * cos(Z) * exp(-0.059 / cos(Z))  for cos(Z)>0, else 0.

## 3. Plane-of-array (POA)  (`physics.poa_global`, `physics.cos_aoi`)  [IMPLEMENTED]
Irradiance is transposed onto the tilted panel plane (tilt, azimuth) with an **isotropic sky model**:
- beam:     `DNI * max(cos(AOI), 0)`,  where `DNI = direct_horizontal / cos(Z)`
- diffuse:  `DHI * (1 + cos(tilt)) / 2`
- ground:   `GHI * albedo * (1 - cos(tilt)) / 2`   (albedo ~ 0.2)
- `POA = beam + diffuse + ground`  (clipped >= 0)

with `cos(AOI) = cos(Z)cos(tilt) + sin(Z)sin(tilt)cos(solar_azimuth - surface_azimuth)`.
Both `clear_sky_power` and `observed_power` feed **POA** (not raw GHI) into the PV model. If pvlib is
installed, `pvlib.irradiance.get_total_irradiance(...)` is an alternative; the built-in isotropic model is the default.

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
clear_sky_ghi(site, t)  ->  poa_global(site, t, ghi, direct, dhi)  ->  pv_power(poa, temp, site)  ->  clear_sky_power  (the prior the ML corrects)
