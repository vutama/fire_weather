#!/usr/bin/env python
"""
fwdp.py

Fire Weather Diagnostics Package

Contains primary functions for computing fire weather variables.

Project Lead: Sebastian Utama (1)
Developer: Cameron Cummins (2)
Contacts: vincentius.utama@utexas.edu (1), cameron.cummins@utexas.edu (2)
2/27/24
"""
import numpy as np
from numba import njit, prange
import xarray


def calcDayLength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0 + dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle / 15.0


def calcEffectiveDayLengths(num_doy, lat_values):
    effective_day_lengths = np.full(fill_value=0.0, shape=(num_doy, lat_values.size), dtype=lat_values.dtype)
    for doy in range(effective_day_lengths.shape[0]):
        for i in range(effective_day_lengths.shape[1]):
            effective_length = calcDayLength(doy + 1, lat_values[i]) - 3.0
            if effective_length < 0.0:
                effective_length = 0.0

            effective_day_lengths[doy, i] = effective_length
    return effective_day_lengths


def calcEffectiveGlobalDayLengths(num_doy, lat_values):
    effective_day_lengths = np.full(fill_value=-999.0, shape=(num_doy, lat_values.size), dtype=lat_values.dtype)
    for doy in range(effective_day_lengths.shape[0]):
        for i in range(effective_day_lengths.shape[1]):
            effective_length = calcDayLength(doy + 1, lat_values[i]) - 3
            if effective_length < 0:
                effective_length = 0

            effective_length = 1.43*effective_length - 4.25
            if effective_length < -1.6:
                effective_length = -1.6

            effective_day_lengths[doy, i] = effective_length
    return effective_day_lengths


@njit(parallel=True)
def calcISI(FFMC, sfcwind):
    ISI = np.full(fill_value=-999.0, shape=FFMC.shape, dtype=FFMC.dtype)

    for i in prange(ISI.shape[1]):
        for j in prange(ISI.shape[2]):
            for t in prange(ISI.shape[0]):
                FW = np.exp(0.05039*sfcwind[t, i, j])
                m = 147.2 * (101.0 - FFMC[t, i, j]) / (59.5 + FFMC[t, i, j])
                FF = 91.9*np.exp(-0.1386*m)*(1.0 + ((m**5.31) / (4.93*10**7)))

                ISI[t, i, j] = 0.208*FW*FF
    return ISI


@njit(parallel=True)
def calcBUI(DMC, DC):
    BUI = np.full(fill_value=0.0, shape=DMC.shape, dtype=DMC.dtype)
    for i in prange(0, BUI.shape[1]):
        for j in prange(0, BUI.shape[2]):
            for t in range(0, BUI.shape[0]):

                dmc = DMC[t, i, j]
                dc = DC[t, i, j]
                if dc < 0:
                    continue
                elif 0 < dmc <= (0.4*dc):
                    BUI[t, i, j] = (0.8*dmc*dc) / (dmc + 0.4*dc)
                elif 0.4*dc < dmc:
                    BUI[t, i, j] = dmc - (1 - ((0.8*dc) / (dmc + 0.4*dc))) * (0.92 + (0.0114*dmc)**1.7)
    return BUI


@njit(parallel=True)
def calcDC(tasmax, pr, effective_global_day_lengths, ndays=365):
    dc0 = np.full(fill_value=15.0, shape=tasmax.shape[1:], dtype=tasmax.dtype)
    DC = np.full(fill_value=-999.0, shape=tasmax.shape, dtype=tasmax.dtype)

    for i in prange(tasmax.shape[1]):
        for j in prange(tasmax.shape[2]):
            for t in range(tasmax.shape[0]):
                doy = t % ndays

                effective_rainfall = 0.83*pr[t, i, j] - 1.27
                if pr[t, i, j] <= 2.8:
                    effective_rainfall = 0.0

                q_r = 800.0*np.exp((-1*dc0[i, j])/400.0) + 3.937*effective_rainfall
                drought_code_rain = 400.0*np.log(800.0/q_r)
                if drought_code_rain <= 0:
                    drought_code_rain = 0.0

                drought_code_dry = 0.5*(0.36*(tasmax[t, i, j] + 2.8) + effective_global_day_lengths[doy, i])
                if tasmax[t, i, j] <= -2.8:
                    drought_code_dry = 0.5*effective_global_day_lengths[doy, i]

                DC[t, i, j] = drought_code_rain + drought_code_dry
                dc0[i, j] = drought_code_rain + drought_code_dry
    return DC


@njit(parallel=True)
def calcDMC(tasmax, pr, hurs, effective_day_lengths, ndays=365):
    initial_DMC = np.full(fill_value=50.0, shape=tasmax.shape[1:], dtype=tasmax.dtype)
    init_moisture_content = 20.0 + np.exp(5.6348 - initial_DMC/43.43)

    DMC = np.full(fill_value=-999.0, shape=tasmax.shape, dtype=tasmax.dtype)

    for i in prange(DMC.shape[1]):
        for j in prange(DMC.shape[2]):
            for t in range(DMC.shape[0]):
                doy = t % ndays

                effective_rainfall = 0.0
                if pr[t, i, j] > 1.5:
                    effective_rainfall = (0.92*pr[t, i, j]) - 1.27

                b = -999.0
                init_dmc = initial_DMC[i, j]
                if init_dmc <= 33:
                    b = 100.0 / (0.5 + 0.3*init_dmc)
                elif 33 < init_dmc <= 65:
                    b = 14.0 - 1.3*np.log(init_dmc)
                elif 65 < init_dmc:
                    b = 6.2*np.log(init_dmc) - 17.2

                moisture_content = init_moisture_content[i, j] + (1000.0*effective_rainfall) / (48.77 + b*effective_rainfall)

                DMC_wet = 244.72 - 43.43*np.log(moisture_content - 20)
                if DMC_wet < 0:
                    DMC_wet = 0.0

                DMC_dry = 0.0
                if tasmax[t, i, j] >= -1.1:
                    DMC_dry = 1.894*(tasmax[t, i, j] + 1.1) * (100.0 - hurs[t, i, j]) * effective_day_lengths[doy, i]*10**(-4)

                DMC[t, i, j] = DMC_wet + DMC_dry
                init_moisture_content[i, j] = 20.0 + np.exp(5.6348 - DMC[t, i, j]/43.43)
                initial_DMC[i, j] = DMC[t, i, j]
    return DMC


@njit
def stepFFMC(tasmax, pr, hurs, sfcwind, moist_content_init):
    if hurs >= 100:
        return (100, 147.2 * (101.0 - 100) / (59.5 + 100))

    effective_rainfall = pr - 0.5

    if effective_rainfall < 0:
        effective_rainfall = 0.0

    dry_equil_moist_content = 0.942*(hurs**0.679) + 11*np.exp((hurs - 100.0)/10.0) + 0.18*(21.1 - (tasmax)) * (1.0 - np.exp(-0.115*hurs))

    wet_equil_moist_content = 0.618*(hurs**0.753) + 10*np.exp((hurs - 100.0)/10.0) + 0.18*(21.1 - (tasmax))*(1.0 - np.exp(-0.115*hurs))

    exp_coeff = 0.0
    if effective_rainfall != 0:
        exp_coeff = np.exp(-(6.93/effective_rainfall)) # problematic exponential
    moist_content_r_small = moist_content_init + (42.5*effective_rainfall*np.exp(-100.0/(251.0 - moist_content_init))*(1.0 - exp_coeff))

    moist_content_r = 0.0
    if 0 < moist_content_init <= 150:
        moist_content_r = moist_content_r_small
    elif 150 < moist_content_init <= 250:
        moist_content_r = moist_content_r_small + (0.0015*(moist_content_init - 150.0)**2) * np.sqrt(effective_rainfall)
    elif 250 < moist_content_init:
        moist_content_r = 250.0

    moist_content = -999.0
    if moist_content_r < dry_equil_moist_content and moist_content_r > wet_equil_moist_content:
        moist_content = moist_content_r

    elif moist_content_r > dry_equil_moist_content:
        dry_diffuse_rate = 0.581*np.exp(0.0365*(tasmax)) * (0.424*(1-(hurs/100.0)**1.7) + 0.0694*np.sqrt(sfcwind) * (1.0 - (hurs/100.0)**8))
        moist_content = dry_equil_moist_content + (moist_content_r - dry_equil_moist_content) * (10**(-1*dry_diffuse_rate))

    elif moist_content_r < wet_equil_moist_content:
        wet_diffuse_rate = 0.581*np.exp(0.0365*(tasmax)) * (0.424*(1 - ((100.0 - hurs)/100.0)**1.7) + 0.0694*np.sqrt(sfcwind) * (1.0 - ((100.0 - hurs)/100.0)**8))
        moist_content = wet_equil_moist_content - (wet_equil_moist_content - moist_content_r) * (10**(-1*wet_diffuse_rate))

    ffmc = 59.5 * (250 - moist_content) / (147.2 + moist_content)
    moist_content_init = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)

    return (ffmc, moist_content_init)


@njit(parallel=True)
def calcFFMC(tasmax, pr, hurs, sfcwind):
    FFMC = np.full(fill_value=0.0, shape=tasmax.shape, dtype=tasmax.dtype)
    moist_content_init = np.full(fill_value=125.0, shape=FFMC.shape[1:], dtype=tasmax.dtype)

    for i in prange(FFMC.shape[1]):
        for j in prange(FFMC.shape[2]):
            for t in range(FFMC.shape[0]):
                ffmc, moist_content_init_i = stepFFMC(
                    tasmax[t, i, j],
                    pr[t, i, j],
                    hurs[t, i, j],
                    sfcwind[t, i, j],
                    moist_content_init[i, j]
                )
                FFMC[t, i, j] = ffmc
                moist_content_init[i, j] = moist_content_init_i

    return FFMC


@njit(parallel=True)
def calcFWI(ISI, BUI):
    FWI = np.full(fill_value=-999.0, shape=ISI.shape, dtype=ISI.dtype)

    for i in prange(FWI.shape[1]):
        for j in prange(FWI.shape[2]):
            for t in prange(FWI.shape[0]):
                Fd = -999.0

                if BUI[t, i, j] <= 80:
                    Fd = 0.626*BUI[t, i, j]**0.809 + 2.0
                else:
                    Fd = 1000.0 / (25.0 + 108.64*np.exp(-0.023*BUI[t, i, j]))

                B = 0.1*Fd*ISI[t, i, j]

                if B >= 1:
                    FWI[t, i, j] = np.exp(2.72*(0.434*np.log(B))**0.647)
                else:
                    FWI[t, i, j] = B

    return FWI


def computeFireWeatherIndices(TMAX: xarray.DataArray, PR: xarray.DataArray, RH: xarray.DataArray, WIND: xarray.DataArray, ndays=365) -> xarray.Dataset:
    TMAX = TMAX.astype(np.double)
    RH = RH.astype(np.double)
    PR = PR.astype(np.double)
    WIND = WIND.astype(np.double)
    TMAX['lat'] = TMAX.lat.astype(np.double)
    
    assert TMAX.dtype == PR.dtype
    assert TMAX.dtype == RH.dtype
    assert TMAX.dtype == WIND.dtype
    assert TMAX.dtype == TMAX.lat.dtype
    assert TMAX.shape == PR.shape
    assert TMAX.shape == RH.shape
    assert TMAX.shape == WIND.shape

    lat_vals = TMAX.lat.values
    eff_day_len = calcEffectiveDayLengths(ndays, lat_vals)
    eff_global_day_len = calcEffectiveGlobalDayLengths(ndays, lat_vals)

    ffmc = calcFFMC(TMAX.values, PR.values, RH.values, WIND.values)
    dmc = calcDMC(TMAX.values, PR.values, RH.values, eff_day_len)
    dc = calcDC(TMAX.values, PR.values, eff_global_day_len)

    isi = calcISI(ffmc, WIND.values)
    bui = calcBUI(dmc, dc)

    fwi = calcFWI(isi, bui)

    return xarray.Dataset(data_vars={
                "FFMC": (["time", "lat", "lon"], ffmc),
                "DMC": (["time", "lat", "lon"], dmc),
                "DC": (["time", "lat", "lon"], dc),
                "ISI": (["time", "lat", "lon"], isi),
                "BUI": (["time", "lat", "lon"], bui),
                "FWI": (["time", "lat", "lon"], fwi),
            }, coords=TMAX.coords)
    