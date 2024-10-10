#!/usr/bin/env python
# coding: utf-8

# In[171]:


# -----------------------------------------------------------
# This code snippet loads the ATom observations,
# then samples the FV3 (C96 and C384) along the flight tracks
# thes runs some evaluation
# Contact: Siyuan Wang (siyuan.wang@noaa.gov)
# -----------------------------------------------------------
from netCDF4 import Dataset

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerErrorbar

from datetime import date, datetime, timedelta
import time

from scipy import stats 
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

import os
import fnmatch
import glob
import gc

#plt.style.use('bmh')
plt.style.use('seaborn')

DataDump = '/path_to_data/'

# ------------------------------------------------------------
# this extracts 4d variables (time,lev,lat,lon) from all tiles
# output: 3d array: (time,lev,grid_1d)
# ------------------------------------------------------------
def extract_3d_var(tiles_in, varname):
    var_dims = tiles_in[0][varname].shape
#     out_array = np.zeros((var_dims[0], var_dims[1], len(tiles_in)*var_dims[2]*var_dims[2]))
    out_array = np.zeros((var_dims[0], var_dims[1], len(tiles_in)*var_dims[2]*var_dims[3]))
    for i in range(var_dims[0]):
        var_temp = [np.array(t[varname][i,:,:,:]) for t in tiles_in]
        for j in range(var_dims[1]):
            var_ravel_temp = np.array([])
            for k in range(len(tiles_in)): var_ravel_temp = np.concatenate((var_ravel_temp, var_temp[k][j,:,:].ravel()))
            out_array[i,j,:] = var_ravel_temp
            del var_ravel_temp
        del var_temp
    return out_array

def extract_2d_var(tiles_in, varname):
    var_dims = tiles_in[0][varname].shape
    out_array = np.zeros((var_dims[0], len(tiles_in)*var_dims[1]*var_dims[2]))
    for i in range(var_dims[0]):
        var_temp = [np.array(t[varname][i,:,:]) for t in tiles_in]
        var_ravel_temp = np.array([])
        for k in range(len(tiles_in)): var_ravel_temp = np.concatenate((var_ravel_temp, var_temp[k][:,:].ravel()))
        out_array[i,:] = var_ravel_temp
        del var_ravel_temp
        del var_temp
    return out_array

def yyyyddd_to_yyyymmdd(year_in, doy_in):
    month = (datetime(year_in, 1, 1) + timedelta(doy_in - 1)).month
    day = (datetime(year_in, 1, 1) + timedelta(doy_in - 1)).day
    outstr = '{0:04d}'.format(year_in) + '{0:02d}'.format(month) + '{0:02d}'.format(day)
    return outstr

def fv3_var_tile(fv3_dir_in, fv3_wildcard_in):
    fv3_var_filelist_temp = [fv3_dir_in+"/"+f for f in sorted(os.listdir(fv3_dir_in)) if fnmatch.fnmatch(f, fv3_wildcard_in)]
    return [Dataset(f) for f in fv3_var_filelist_temp]


# In[172]:


meas_aod_file = DataDump+'Brock2021_CompreAerosol_ATOM/ATom_Aerosol_Properties_1908/data/ATom_aerosol_profiles.nc'
meas_aod = xr.open_dataset(meas_aod_file)

meas_aod['year'] = meas_aod['Mid_Date_Time_UTC.year']
meas_aod['month'] = meas_aod['Mid_Date_Time_UTC.month']
meas_aod['day'] = meas_aod['Mid_Date_Time_UTC.day']
meas_aod['hour'] = meas_aod['Mid_Date_Time_UTC.hour']
meas_aod['minute'] = meas_aod['Mid_Date_Time_UTC.minute']
meas_aod['second'] = meas_aod['Mid_Date_Time_UTC.second']

meas_aod['start_year'] = meas_aod['Start_Date_Time_UTC.year']
meas_aod['start_month'] = meas_aod['Start_Date_Time_UTC.month']
meas_aod['start_day'] = meas_aod['Start_Date_Time_UTC.day']
meas_aod['start_hour'] = meas_aod['Start_Date_Time_UTC.hour']
meas_aod['start_minute'] = meas_aod['Start_Date_Time_UTC.minute']
meas_aod['start_second'] = meas_aod['Start_Date_Time_UTC.second']

meas_aod['end_year'] = meas_aod['End_Date_Time_UTC.year']
meas_aod['end_month'] = meas_aod['End_Date_Time_UTC.month']
meas_aod['end_day'] = meas_aod['End_Date_Time_UTC.day']
meas_aod['end_hour'] = meas_aod['End_Date_Time_UTC.hour']
meas_aod['end_minute'] = meas_aod['End_Date_Time_UTC.minute']
meas_aod['end_second'] = meas_aod['End_Date_Time_UTC.second']

meas_aod['mid_longitude_0to360'] = (('Mid_Date_Time_UTC'),np.where(meas_aod['mid_longitude']<0., 360.+meas_aod['mid_longitude'], meas_aod['mid_longitude']))
meas_aod['start_longitude_0to360'] = (('Mid_Date_Time_UTC'),np.where(meas_aod['start_longitude']<0., 360.+meas_aod['start_longitude'], meas_aod['start_longitude']))
meas_aod['end_longitude_0to360'] = (('Mid_Date_Time_UTC'),np.where(meas_aod['end_longitude']<0., 360.+meas_aod['end_longitude'], meas_aod['end_longitude']))

# --- ind for 550nm AOD
ind_aod550 = 5

# --- add deployent id: ATom1-ATom4
deployment = np.where(meas_aod['year']==2016, 'ATom1',
                      np.where((meas_aod['year']==2017) & (meas_aod['month']<=2), 'ATom2', 
                               np.where((meas_aod['year']==2017) & (meas_aod['month']>=8), 'ATom3', 
                                        np.where(meas_aod['year']==2018, 'ATom4', 'huh?'))))
meas_aod['deployment'] = (('Mid_Date_Time_UTC'), deployment)

# --- calculate fractions
meas_aod['frac_alk'] = meas_aod['tau_dry_alkali_salts']/meas_aod['tau']
meas_aod['frac_BB'] = meas_aod['tau_biomass_burning']/meas_aod['tau']
meas_aod['frac_BC'] = meas_aod['tau_abs_BC']/meas_aod['tau']
meas_aod['frac_dust'] = meas_aod['tau_dust']/meas_aod['tau']
meas_aod['frac_met'] = meas_aod['tau_meteoric']/meas_aod['tau']
meas_aod['frac_SON'] = meas_aod['tau_sulfate_organic']/meas_aod['tau']
meas_aod['frac_combustion'] = meas_aod['tau_combustion']/meas_aod['tau']
meas_aod['frac_SS'] = meas_aod['tau_sea_salt']/meas_aod['tau']
meas_aod['density_kg_m3'] = 28.97/1000./8.314 * (100.*meas_aod['ambient_pressure']) / meas_aod['ambient_temperature']
meas_aod['STP_to_AMB'] = meas_aod['ambient_pressure'] * 273.15 / 1013.25 / meas_aod['ambient_temperature']

# --- layer thickness
LayerThickness_m = np.zeros(len(meas_aod['mid_altitude']))
for i in range(len(meas_aod['mid_altitude'])):
    if i==0: LayerThickness_m[i] = 2*meas_aod['mid_altitude'][i]
    else: LayerThickness_m[i] = 2*(meas_aod['mid_altitude'][i]-meas_aod['mid_altitude'][i-1]-0.5*LayerThickness_m[i-1])

def col_density(in_var):
    return np.nansum(meas_aod[in_var]*LayerThickness_m,axis=1)

# --- convert mass concentrations from STP to ambient
meas_aod['Sulfate_ug_m3'] = (meas_aod['Sulfate_fine'] + meas_aod['Sulfate_coarse']) * meas_aod['STP_to_AMB']
meas_aod['Nitrate_ug_m3'] = meas_aod['Nitrate_fine'] * meas_aod['STP_to_AMB']
meas_aod['OA_ug_m3'] = (meas_aod['OA_fine']) * meas_aod['STP_to_AMB']
meas_aod['Sea_Salt_ug_m3'] = (meas_aod['Sea_Salt_fine'] + meas_aod['Sea_Salt_coarse']) * meas_aod['STP_to_AMB']
meas_aod['Dust_ug_m3'] = (meas_aod['Dust_fine'] + meas_aod['Dust_coarse']) * meas_aod['STP_to_AMB']
meas_aod['BC_SP2_ug_m3'] = meas_aod['BC_SP2'] * meas_aod['STP_to_AMB']

# --- calculate column density
meas_aod['col_Sea_Salt_ug_m2'] = (('Mid_Date_Time_UTC'),col_density('Sea_Salt_ug_m3'))
meas_aod['col_Dust_ug_m2'] = (('Mid_Date_Time_UTC'),col_density('Dust_ug_m3'))
meas_aod['col_OA_ug_m2'] = (('Mid_Date_Time_UTC'),col_density('OA_ug_m3'))
meas_aod['col_Nitrate_ug_m2'] = (('Mid_Date_Time_UTC'),col_density('Nitrate_ug_m3'))
meas_aod['col_Sulfate_ug_m2'] = (('Mid_Date_Time_UTC'),col_density('Sulfate_ug_m3'))
meas_aod['col_BC_ug_m2'] = (('Mid_Date_Time_UTC'),col_density('BC_SP2_ug_m3'))

# --- measurement uncertainty of AOD
meas_aod['tau_err'] = meas_aod['tau']*0.3

# --- create different ATom deployments
# meas_aod_ATom1 = meas_aod.where(meas_aod['deployment']=='ATom1',drop=True)
# meas_aod_ATom2 = meas_aod.where(meas_aod['deployment']=='ATom2',drop=True)
# meas_aod_ATom3 = meas_aod.where(meas_aod['deployment']=='ATom3',drop=True)
meas_aod_ATom4 = meas_aod.where(meas_aod['deployment']=='ATom4',drop=True)

meas_aod_ATom4


# In[183]:


# --- this one is updated: crop early on, i.e., when extracting the variables
#     will double the speed. but still slow AF for C384

def extract_3d_var_full(tiles_in, varname):
    var_dims = tiles_in[0][varname].shape
    out_array = np.zeros((var_dims[0], var_dims[1], len(tiles_in)*var_dims[2]*var_dims[3]))
    for i in range(var_dims[0]):
        var_temp = [np.array(t[varname][i,:,:,:]) for t in tiles_in]
        for j in range(var_dims[1]):
            var_ravel_temp = np.array([])
            for k in range(len(tiles_in)): var_ravel_temp = np.concatenate((var_ravel_temp, var_temp[k][j,:,:].ravel()))
            out_array[i,j,:] = var_ravel_temp
            del var_ravel_temp
        del var_temp
    return out_array

def extract_2d_var_full(tiles_in, varname):
    var_dims = tiles_in[0][varname].shape
    out_array = np.zeros((var_dims[0], len(tiles_in)*var_dims[1]*var_dims[2]))
    for i in range(var_dims[0]):
        var_temp = [np.array(t[varname][i,:,:]) for t in tiles_in]
        var_ravel_temp = np.array([])
        for k in range(len(tiles_in)): var_ravel_temp = np.concatenate((var_ravel_temp, var_temp[k][:,:].ravel()))
        out_array[i,:] = var_ravel_temp
        del var_ravel_temp
        del var_temp
    return out_array

def extract_pressure_full(core_res_tiles_in):
    # --- crop here. will make things faster!!!
    T_K_temp = extract_3d_var_full(core_res_tiles_in, 'T')[0,:,:]     # temperature in K
    delp_temp = extract_3d_var_full(core_res_tiles_in, 'delp')[0,:,:] # dP in Pa
    DZ_temp = extract_3d_var_full(core_res_tiles_in, 'DZ')[0,:,:]     # negative dZ in meter
    phis_temp = extract_2d_var_full(core_res_tiles_in, 'phis')[0,:] # surface geopotential in m2/s2
    Pressure_Pa_temp = T_K_temp.copy()
    LayerThickness_m_temp = T_K_temp.copy()
    # --- make array
    z_layer_interface_m_temp = np.ndarray(shape=(1+T_K_temp.shape[0], T_K_temp.shape[1]), dtype=float, order='F')
    z_layer_interface_m_temp[-1+z_layer_interface_m_temp.shape[0],:] = phis_temp/9.81
    for j in range(-1+LayerThickness_m_temp.shape[0],-1,-1):
        z_layer_interface_m_temp[j,:] = z_layer_interface_m_temp[j+1,:] - DZ_temp[j,:]
        LayerThickness_m_temp[j,:] = z_layer_interface_m_temp[j,:]-z_layer_interface_m_temp[j+1,:]
    # --- hydrostatic equation
    Pressure_Pa_out = 286.9 * T_K_temp / 9.81 * delp_temp / LayerThickness_m_temp
    del delp_temp, DZ_temp, phis_temp
    return Pressure_Pa_out, T_K_temp

def extract_3d_var(tiles_in, varname, grid_latt_patch_in):
    var_dims = tiles_in[0][varname].shape
    out_array = np.zeros((var_dims[0], var_dims[1], np.count_nonzero(~np.isnan(grid_latt_patch_in))))
    for i in range(var_dims[0]):
        var_temp = [np.array(t[varname][i,:,:,:]) for t in tiles_in]
        for j in range(var_dims[1]):
            var_ravel_temp = np.array([])
            for k in range(len(tiles_in)): var_ravel_temp = np.concatenate((var_ravel_temp, var_temp[k][j,:,:].ravel()))
            out_array[i,j,:] = var_ravel_temp[~np.isnan(grid_latt_patch_in)].astype(np.float32,copy=False)
            del var_ravel_temp
        del var_temp
    # gc.collect()
    return out_array

def extract_2d_var(tiles_in, varname, grid_latt_patch_in):
    var_dims = tiles_in[0][varname].shape
    out_array = np.zeros((var_dims[0], np.count_nonzero(~np.isnan(grid_latt_patch_in))))
    for i in range(var_dims[0]):
        var_temp = [np.array(t[varname][i,:,:]) for t in tiles_in]
        var_ravel_temp = np.array([])
        for k in range(len(tiles_in)): var_ravel_temp = np.concatenate((var_ravel_temp, var_temp[k][:,:].ravel()))
        out_array[i,:] = var_ravel_temp[~np.isnan(grid_latt_patch_in)].astype(np.float32,copy=False)
        del var_ravel_temp
        del var_temp
    # gc.collect()
    return out_array

def fv3_var_tile(fv3_dir_in, fv3_wildcard_in, tile_id_in):
    fv3_var_filelist_temp = [fv3_dir_in+"/"+f for f in sorted(os.listdir(fv3_dir_in)) if fnmatch.fnmatch(f, fv3_wildcard_in)]
    fv3_var_filelist_temp = [fv3_var_filelist_temp[n] for n in tile_id_in]
    return [Dataset(f) for f in fv3_var_filelist_temp]

def interp_lev_loc(tri_in, tiles_in, var2load_in, lev_ind_in, intp_lon_in,intp_lat_in, grid_latt_patch_in):
    var_temp = [extract_3d_var(tiles_in, v, grid_latt_patch_in)[0,:,:] for v in var2load_in]  # <- crop here
    interpolator_temp = [LinearNDInterpolator(tri_in, v[lev_ind_in,:]) for v in var_temp] 
    var_intp_temp = [iph( (intp_lon_in,intp_lat_in) ) for iph in interpolator_temp]
    del var_temp,interpolator_temp
    return var_intp_temp

def interp_pressure_loc(tri_in, core_res_tiles_in, intp_lon_in,intp_lat_in, grid_latt_patch_in):
    pressure_temp, T_K_temp = extract_pressure(core_res_tiles_in, grid_latt_patch_in)  # <- already cropped
    interpolator_temp = [LinearNDInterpolator(tri_in, pressure_temp[lev_ind,:]) for lev_ind in range(pressure_temp.shape[0])] # <- crop here
    pressure_intp_temp = [iph( (intp_lon_in,intp_lat_in) ) for iph in interpolator_temp]
    interpolator_T_temp = [LinearNDInterpolator(tri_in, T_K_temp[lev_ind,:]) for lev_ind in range(T_K_temp.shape[0])] # <- crop here
    T_intp_temp = [iph( (intp_lon_in,intp_lat_in) ) for iph in interpolator_T_temp]
    return pressure_intp_temp, T_intp_temp

def extract_pressure(core_res_tiles_in, grid_latt_patch_in):
    # --- crop here. will make things faster!!!
    # T_K_temp = extract_3d_var(core_res_tiles_in, 'T')[0,:,:][:,~np.isnan(grid_latt_patch)]     # temperature in K
    # delp_temp = extract_3d_var(core_res_tiles_in, 'delp')[0,:,:][:,~np.isnan(grid_latt_patch)] # dP in Pa
    # DZ_temp = extract_3d_var(core_res_tiles_in, 'DZ')[0,:,:][:,~np.isnan(grid_latt_patch)]     # negative dZ in meter
    # phis_temp = extract_2d_var(core_res_tiles_in, 'phis')[0,:][~np.isnan(grid_latt_patch)] # surface geopotential in m2/s2
    T_K_temp = extract_3d_var(core_res_tiles_in, 'T', grid_latt_patch_in)[0,:,:]     # temperature in K
    delp_temp = extract_3d_var(core_res_tiles_in, 'delp', grid_latt_patch_in)[0,:,:] # dP in Pa
    DZ_temp = extract_3d_var(core_res_tiles_in, 'DZ', grid_latt_patch_in)[0,:,:]     # negative dZ in meter
    phis_temp = extract_2d_var(core_res_tiles_in, 'phis', grid_latt_patch_in)[0,:] # surface geopotential in m2/s2
    Pressure_Pa_temp = T_K_temp.copy()
    LayerThickness_m_temp = T_K_temp.copy()
    #####z_layer_center_m_temp = T_K_temp.copy()
    # --- make array
    z_layer_interface_m_temp = np.ndarray(shape=(1+T_K_temp.shape[0], T_K_temp.shape[1]), dtype=float, order='F')
    z_layer_interface_m_temp[-1+z_layer_interface_m_temp.shape[0],:] = phis_temp/9.81
    for j in range(-1+LayerThickness_m_temp.shape[0],-1,-1):
        z_layer_interface_m_temp[j,:] = z_layer_interface_m_temp[j+1,:] - DZ_temp[j,:]
        LayerThickness_m_temp[j,:] = z_layer_interface_m_temp[j,:]-z_layer_interface_m_temp[j+1,:]
        #####z_layer_center_m_temp[j,:] = 0.5*(z_layer_interface_m_temp[j,:]+z_layer_interface_m_temp[j+1,:])
    # --- hydrostatic equation
    Pressure_Pa_out = 286.9 * T_K_temp / 9.81 * delp_temp / LayerThickness_m_temp
    del delp_temp, DZ_temp, phis_temp
    return Pressure_Pa_out, T_K_temp

def interp_to_press_grid(press_grid_in, press_in):
    if (press_in>=np.nanmax(press_grid_in)):
        grid_ind_lower = np.argmax(press_grid_in)
        grid_ind_upper = np.argmax(press_grid_in)
        frac_from_lower = 0.
    else:
        closest_ind = np.argmin(abs(np.array(press_grid_in)-press_in))
        if (press_grid_in[closest_ind]>=press_in):
            grid_ind_upper = closest_ind
            grid_ind_lower = closest_ind - 1
        else:
            grid_ind_upper = closest_ind + 1
            grid_ind_lower = closest_ind
        grid_ind_lower = max([0,grid_ind_lower])
        grid_ind_upper = min([-1+len(press_grid_in),grid_ind_upper])
        frac_from_lower = (press_in - press_grid_in[grid_ind_lower])/(press_grid_in[grid_ind_upper] - press_grid_in[grid_ind_lower])
    return grid_ind_lower, grid_ind_upper, frac_from_lower

def interp_var_to_lev(frac_from_lower_in, val_lower_in, val_upper_in):
    return val_lower_in*(1.-frac_from_lower_in) + val_upper_in*frac_from_lower_in









var2load = ['maod','maodbc','maoddust','maodoc','maodseas','maodsulf']

# --- C384
fv3_dir_base = '/scratch2/BMC/wrfruc/Shan.Sun/wgne_chem_exp/COMROOT/vari_aer2/gfs.20180501/00/atmos/'
fv3_outputfile_keyword = 'sfcf*.tile*.nc'

fv3_hours_since = [fv3_dir_base+f for f in sorted(os.listdir(fv3_dir_base)) if fnmatch.fnmatch(f, fv3_outputfile_keyword)]
fv3_hours_since = [int(f.split('/')[-1].split('.')[0].replace('sfcf','')) for f in fv3_hours_since]
fv3_hours_since = np.unique(fv3_hours_since)

buffer_deg = 0.5  # 2.  # can't be too small. doesn't affect speed anyway.

grid_spec_dir = '/scratch1/NCEPDEV/global/glopara/fix/fix_fv3/C384/'
grid_spec_wildcard = 'C384_grid_spec.tile*.nc'

grid_spec_filelist = [grid_spec_dir+f for f in sorted(os.listdir(grid_spec_dir)) if fnmatch.fnmatch(f, grid_spec_wildcard)]
grid_tiles = [Dataset(f) for f in grid_spec_filelist]

# ----------------------
# load the grid: raveled
# ----------------------
grid_latt_tiles = [np.array(t['grid_latt']) for t in grid_tiles]
grid_lont_tiles = [np.array(t['grid_lont']) for t in grid_tiles]
grid_lont_tiles = [np.where(lon<180., lon, lon-360.) for lon in grid_lont_tiles]
grid_latt = np.array(grid_latt_tiles).ravel()
grid_lont = np.array(grid_lont_tiles).ravel()
grid_tile_id = np.array([np.ones(grid_latt_tiles[n].shape) * n for n in range(len(grid_latt_tiles))]).ravel()

in_meas_aod = meas_aod_ATom4.copy()

tile_id_previous_step = []
hour_previous_step = -999
n_rec = in_meas_aod['Mid_Date_Time_UTC'].shape[0]

housekeeping = np.empty((n_rec,8)) * np.nan
var_extracted = np.empty((n_rec,len(var2load))) * np.nan
press_temp_extracted = np.empty((n_rec,2)) * np.nan



for i in range(n_rec):
    gc.collect()
    # start_time = time.time()
    # ----------------------------
    # load the time and coordiates
    # ----------------------------
    intp_lon, intp_lat = in_meas_aod['mid_longitude'].values[i],in_meas_aod['mid_latitude'].values[i] 
    intp_year,intp_month,intp_day = int(in_meas_aod['year'].values[i]),int(in_meas_aod['month'].values[i]),int(in_meas_aod['day'].values[i])
    intp_hour,intp_minute,intp_sec = int(in_meas_aod['hour'].values[i]),int(in_meas_aod['minute'].values[i]),int(in_meas_aod['second'].values[i])
    dt = datetime(intp_year,intp_month,intp_day,intp_hour,intp_minute,intp_sec)-datetime(intp_year,5,1,0,0)
    hours_since = dt.days*24. + dt.seconds/3600.
    housekeeping[i,:] = [intp_year,intp_month,intp_day,intp_hour,intp_minute,intp_sec,intp_lon,intp_lat]
    print( '--- %d/%d: %4d-%02d-%02d %02d:%02d:%02d' % (i+1,n_rec,intp_year,intp_month,intp_day,intp_hour,intp_minute,intp_sec))
    print( '    ', intp_lon,intp_lat )
    if (np.isnan(housekeeping[i,:]).any()) | (intp_month==4):
        print( '   *** bad input or beyond date range *** ' )
    else:
        # ------------------------------------------------------
        # get the file pointers: previous and next step (3-hour)
        # ------------------------------------------------------
        closest_ind = abs(fv3_hours_since-hours_since).argmin()
        if (fv3_hours_since[closest_ind]>hours_since):
            fv3_hour_ind_next = fv3_hours_since[closest_ind]
            fv3_hour_ind_prev = fv3_hours_since[closest_ind-1]
        else:
            fv3_hour_ind_next = fv3_hours_since[closest_ind+1]
            fv3_hour_ind_prev = fv3_hours_since[closest_ind]
        intp_step_frac = (hours_since-fv3_hour_ind_prev)/(fv3_hour_ind_next-fv3_hour_ind_prev)
        print( '    ', hours_since, fv3_hour_ind_prev, fv3_hour_ind_next, intp_step_frac )
        fv3_file_wildcard_Next = 'sfcf%03d.tile*.nc' % (fv3_hour_ind_next)
        fv3_file_wildcard_Prev = 'sfcf%03d.tile*.nc' % (fv3_hour_ind_prev)
        print( '    ', fv3_file_wildcard_Prev, fv3_file_wildcard_Next )
        # -------------------------------
        # which tile(s) is this point in?
        # -------------------------------
        buffer_patch = (grid_latt<intp_lat+buffer_deg) & (grid_latt>intp_lat-buffer_deg) & (grid_lont<intp_lon+buffer_deg) & (grid_lont>intp_lon-buffer_deg)
        grid_latt_patch = np.where(buffer_patch,grid_latt, np.nan)
        grid_lont_patch = np.where(buffer_patch,grid_lont, np.nan)    
        # --- make the cropped mesh 
        if (np.count_nonzero(~np.isnan(grid_lont_patch))<4):  # need at least 4 points
            buffer_patch = (grid_latt<intp_lat+buffer_deg*2.) & (grid_latt>intp_lat-buffer_deg*2.) & (grid_lont<intp_lon+buffer_deg*2.) & (grid_lont>intp_lon-buffer_deg*2.)
            grid_latt_patch = np.where(buffer_patch,grid_latt, np.nan)
            grid_lont_patch = np.where(buffer_patch,grid_lont, np.nan)    
        # --- crop
        grid_latt_crop = grid_latt[~np.isnan(grid_latt_patch)].copy()
        grid_lont_crop = grid_lont[~np.isnan(grid_lont_patch)].copy()
        # --- tile(s) the (lat,lon) point sits!!!
        grid_tile_id_crop = grid_tile_id[~np.isnan(grid_latt_patch)].copy()
        tile_id = [int(f) for f in np.unique(grid_tile_id_crop)] 
        print('    tiles: ',tile_id)
        del grid_latt_patch, grid_lont_patch, buffer_patch
        # -----------------------    
        # crop only a small patch
        # -----------------------
        grid_latt_surr = np.array([grid_latt_tiles[n] for n in tile_id]).ravel()
        grid_lont_surr = np.array([grid_lont_tiles[n] for n in tile_id]).ravel()        
        # --- buffer zone
        buffer_patch = (grid_latt_surr<intp_lat+buffer_deg) & (grid_latt_surr>intp_lat-buffer_deg) & (grid_lont_surr<intp_lon+buffer_deg) & (grid_lont_surr>intp_lon-buffer_deg)
        grid_latt_patch = np.where(buffer_patch,grid_latt_surr, np.nan)
        grid_lont_patch = np.where(buffer_patch,grid_lont_surr, np.nan)  
        print('    N grids in buffer zone: ', np.count_nonzero(~np.isnan(grid_lont_patch)))
        # --- make the cropped mesh 
        if (np.count_nonzero(~np.isnan(grid_lont_patch))<4):  # need at least 4 points
            buffer_patch = (grid_latt_surr<intp_lat+3*buffer_deg) & (grid_latt_surr>intp_lat-3*buffer_deg) & (grid_lont_surr<intp_lon+3*buffer_deg) & (grid_lont_surr>intp_lon-3*buffer_deg)
            grid_latt_patch = np.where(buffer_patch,grid_latt_surr, np.nan)
            grid_lont_patch = np.where(buffer_patch,grid_lont_surr, np.nan)    
        # --- crop and make the Delaunay triangulation
        grid_latt_crop = grid_latt_surr[~np.isnan(grid_latt_patch)].copy()
        grid_lont_crop = grid_lont_surr[~np.isnan(grid_lont_patch)].copy()
        mesh_crop = np.array(list(zip(grid_lont_crop, grid_latt_crop)))
        tri_crop = Delaunay(mesh_crop)
        # ----------------------
        # now load the variables
        # ----------------------
        if (hour_previous_step==fv3_hour_ind_prev) and (tile_id_previous_step==tile_id):
            print( '    no need to load new tiles' )
        else:
            print( '*** load new tiles' )
            # --- load variable tiles
            var_tiles_Prev = fv3_var_tile(fv3_dir_base, fv3_file_wildcard_Prev, tile_id)
            var_tiles_Next = fv3_var_tile(fv3_dir_base, fv3_file_wildcard_Next, tile_id)
            # --- load full fields of all variables
            vars_full_Prev = [extract_2d_var_full(var_tiles_Prev, v)[0,:] for v in var2load]
            vars_full_Next = [extract_2d_var_full(var_tiles_Next, v)[0,:] for v in var2load]
        # --- mow process individual variables
        var_intp = []
        for spc in range(len(var2load)):
            # --- crop
            var_crop_Prev = np.array(vars_full_Prev[spc])[~np.isnan(grid_latt_patch)]
            var_crop_Next = np.array(vars_full_Next[spc])[~np.isnan(grid_latt_patch)]
            # --- interpolate to the exact loc
            intp_var_Prev = LinearNDInterpolator(tri_crop, var_crop_Prev)( (intp_lon,intp_lat) )
            intp_var_Next = LinearNDInterpolator(tri_crop, var_crop_Next)( (intp_lon,intp_lat) )
            # --- interpolate to the exact time stamp
            var_intp.append(interp_var_to_lev(intp_step_frac, intp_var_Prev, intp_var_Next))
        # --- append
        var_extracted[i,:] = np.array(var_intp)
        # --- advance step
        hour_previous_step = fv3_hour_ind_prev
        tile_id_previous_step = tile_id

# --- delete all the useless shits
for f in var_tiles_Prev: f.close()
for f in var_tiles_Next: f.close()
del var_tiles_Prev, var_tiles_Next
del vars_full_Prev, vars_full_Next
del var_crop_Prev, var_crop_Next


# ------------
# housekeeping
# ------------
# --- add col name
var_extracted = pd.DataFrame(var_extracted)
var_extracted.columns = var2load
# --- add housekeeping data
var_extracted['UTC_year'] = [int(f) for f in np.reshape(housekeeping, (len(housekeeping), 8))[:,0]]
var_extracted['UTC_month'] = [int(f) for f in np.reshape(housekeeping, (len(housekeeping), 8))[:,1]]
var_extracted['UTC_day'] = [int(f) for f in np.reshape(housekeeping, (len(housekeeping), 8))[:,2]]
var_extracted['UTC_hour'] = [int(f) for f in np.reshape(housekeeping, (len(housekeeping), 8))[:,3]]
var_extracted['UTC_minute'] = [int(f) for f in np.reshape(housekeeping, (len(housekeeping), 8))[:,4]]
var_extracted['UTC_second'] = [int(f) for f in np.reshape(housekeeping, (len(housekeeping), 8))[:,5]]
var_extracted['UTC_yyyymmdd'] = var_extracted['UTC_year']*10000+var_extracted['UTC_month']*100+var_extracted['UTC_day']
var_extracted['lon'] = np.reshape(housekeeping, (len(housekeeping), 8))[:,6]
var_extracted['lat'] = np.reshape(housekeeping, (len(housekeeping), 8))[:,7]
var_extracted['lon_0to360'] = np.where(var_extracted['lon']<0., 360.+var_extracted['lon'], var_extracted['lon'])


# In[190]:


for v in var2load:
    in_meas_aod[v] = (('Mid_Date_Time_UTC'),var_extracted[v])

in_meas_aod


# In[197]:


plt.plot(in_meas_aod['mid_latitude'], in_meas_aod['tau'][:,ind_aod550],'ko')


# In[271]:


max_range = [1.59, 1.29, 0.39, 0.49, 0.039]
label = ['AOD', 'dust AOD', 'sea salt AOD', 'sulfate+organics\n+smoke AOD', 'black carbon AOD']
d = 'ATom4'
t = 'Apr-May 2018'

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10,12))

for i,ocn in enumerate([(in_meas_aod['mid_longitude_0to360']<=290.), (in_meas_aod['mid_longitude_0to360']>=290.)]):
    # --- pacific vs atlantic
    meas_aod_tmp = in_meas_aod.where(ocn,drop=True)
    for j,(spc_meas,spc_mod) in enumerate(zip(['tau','tau_dust','tau_sea_salt','tau_SONBB','tau_abs_BC'],
                                              ['maod','maoddust','maodseas','maodocsulf','maodbc'])):
        # --- set up measured AOD
        if spc_meas=='tau_SONBB': 
            meas = meas_aod_tmp['tau_sulfate_organic'][:,ind_aod550].values+meas_aod_tmp['tau_biomass_burning'][:,ind_aod550].values
        else:
            meas = meas_aod_tmp[spc_meas][:,ind_aod550].values
        # --- set up modeldd AOD
        if spc_mod=='maodocsulf':
            mod = meas_aod_tmp['maodoc'].values+meas_aod_tmp['maodsulf'].values
        else:
            mod = meas_aod_tmp[spc_mod].values
        # --- plot
        axs[j,i].errorbar(x=meas_aod_tmp['mid_latitude'], y=meas, yerr=meas*0.3,
                          fmt='ks', ms=7, mec='white', mew=1, zorder=200, label='ATom observed')
        axs[j,i].errorbar(x=meas_aod_tmp['mid_latitude'], y=mod,
                          fmt='C2o', ms=7, mec='white', mew=1, zorder=200, label='Modeled')
        # --- misc
        axs[j,i].set_ylim([0,max_range[j]])
        axs[j,i].set_xlim([-75,90])
        axs[j,i].tick_params(axis='both', which='major', labelsize=12)
        if i==0: axs[j,i].set_ylabel(label[j],fontsize=15)

axs[0,0].set_title(d+' ('+t+')'+': Pacific',fontsize=16,fontweight='bold',x=0.02,ha='left',va='top')
axs[0,1].set_title(d+' ('+t+')'+': Atlantic',fontsize=16,fontweight='bold',x=0.02,ha='left',va='top')

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.99, hspace=0.2, wspace=0.1)
axs[-1,0].set_xlabel('Latitude (degree)',fontsize=15)
axs[-1,1].set_xlabel('Latitude (degree)',fontsize=15)
axs[0,0].legend(loc='upper right',frameon=True,facecolor='w')

plt.savefig('mod_AOD_lat_spc_ATom.jpeg', dpi=300)


# In[ ]:




