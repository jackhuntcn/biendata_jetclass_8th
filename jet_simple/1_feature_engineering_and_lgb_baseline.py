import warnings
warnings.simplefilter('ignore')

import gc

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.6f' % x)

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb

# PART 1: loading data

DATA_DIR = './jet_simple_data/'
train = pd.read_csv(DATA_DIR+'simple_train_R04_jet.csv')
test = pd.read_csv(DATA_DIR+'simple_test_R04_jet.csv')

train[['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']].describe()
test[['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']].describe()

# PART 2: feature engineering

def fill_zero(x):
    return 0 if x < 0 else x

train.jet_mass = train.jet_mass.progress_apply(lambda x: fill_zero(x))
test.jet_mass = test.jet_mass.progress_apply(lambda x: fill_zero(x))

def l2_dict(x, y, z): 
    return np.sqrt(np.square(x)+np.square(y)+np.square(z))

train['jet_distance'] = train.progress_apply(lambda row: l2_dict(row.jet_px, row.jet_py, row.jet_pz), axis=1)
test['jet_distance'] = test.progress_apply(lambda row: l2_dict(row.jet_px, row.jet_py, row.jet_pz), axis=1)

train['xy_dis'] = train.progress_apply(lambda row: l2_dict(row.jet_px, row.jet_py, 0), axis=1)
test['xy_dis'] = test.progress_apply(lambda row: l2_dict(row.jet_px, row.jet_py, 0), axis=1)
train['yz_dis'] = train.progress_apply(lambda row: l2_dict(0, row.jet_py, row.jet_pz), axis=1)
test['yz_dis'] = test.progress_apply(lambda row: l2_dict(0, row.jet_py, row.jet_pz), axis=1)
train['zx_dis'] = train.progress_apply(lambda row: l2_dict(row.jet_px, 0, row.jet_pz), axis=1)
test['zx_dis'] = test.progress_apply(lambda row: l2_dict(row.jet_px, 0, row.jet_pz), axis=1)

train['energy_x'] = train['jet_energy'] * train['x_div_dist']
test['energy_x'] = test['jet_energy'] * test['x_div_dist']
train['energy_y'] = train['jet_energy'] * train['y_div_dist']
test['energy_y'] = test['jet_energy'] * test['y_div_dist']
train['energy_z'] = train['jet_energy'] * train['z_div_dist']
test['energy_z'] = test['jet_energy'] * test['z_div_dist']

train['energy_xy'] = train['jet_energy'] * train['xy_div_dist']
test['energy_xy'] = test['jet_energy'] * test['xy_div_dist']
train['energy_yz'] = train['jet_energy'] * train['yz_div_dist']
test['energy_yz'] = test['jet_energy'] * test['yz_div_dist']
train['energy_zx'] = train['jet_energy'] * train['zx_div_dist']
test['energy_zx'] = test['jet_energy'] * test['zx_div_dist']

train['mass_x'] = train['jet_mass'] * train['x_div_dist']
test['mass_x'] = test['jet_mass'] * test['x_div_dist']
train['mass_y'] = train['jet_mass'] * train['y_div_dist']
test['mass_y'] = test['jet_mass'] * test['y_div_dist']
train['mass_z'] = train['jet_mass'] * train['z_div_dist']
test['mass_z'] = test['jet_mass'] * test['z_div_dist']

train['mass_xy'] = train['jet_mass'] * train['xy_div_dist']
test['mass_xy'] = test['jet_mass'] * test['xy_div_dist']
train['mass_yz'] = train['jet_mass'] * train['yz_div_dist']
test['mass_yz'] = test['jet_mass'] * test['yz_div_dist']
train['mass_zx'] = train['jet_mass'] * train['zx_div_dist']
test['mass_zx'] = test['jet_mass'] * test['zx_div_dist']

def angle(x, y):
    return np.degrees(np.math.atan(x / y)) if y != 0 else 0

train['angle_xy'] = train.progress_apply(lambda row: angle(row.jet_px, row.jet_py), axis=1)
test['angle_xy'] = test.progress_apply(lambda row: angle(row.jet_px, row.jet_py), axis=1)
train['angle_yx'] = train.progress_apply(lambda row: angle(row.jet_py, row.jet_px), axis=1)
test['angle_yx'] = test.progress_apply(lambda row: angle(row.jet_py, row.jet_px), axis=1)
train['angle_yz'] = train.progress_apply(lambda row: angle(row.jet_py, row.jet_pz), axis=1)
test['angle_yz'] = test.progress_apply(lambda row: angle(row.jet_py, row.jet_pz), axis=1)
train['angle_zy'] = train.progress_apply(lambda row: angle(row.jet_pz, row.jet_py), axis=1)
test['angle_zy'] = test.progress_apply(lambda row: angle(row.jet_pz, row.jet_py), axis=1)
train['angle_zx'] = train.progress_apply(lambda row: angle(row.jet_pz, row.jet_px), axis=1)
test['angle_zx'] = test.progress_apply(lambda row: angle(row.jet_pz, row.jet_px), axis=1)
train['angle_xz'] = train.progress_apply(lambda row: angle(row.jet_px, row.jet_pz), axis=1)
test['angle_xz'] = test.progress_apply(lambda row: angle(row.jet_px, row.jet_pz), axis=1)

train['mean_particles_mass'] = train['jet_mass'] / train['number_of_particles_in_this_jet']
test['mean_particles_mass'] = test['jet_mass'] / test['number_of_particles_in_this_jet']

train['mean_particles_energy'] = train['jet_energy'] / train['number_of_particles_in_this_jet']
test['mean_particles_energy'] = test['jet_energy'] / test['number_of_particles_in_this_jet']

def calculate_speed(e, m):
    return np.sqrt(2*e/m) if m > 0 else 0

train['jet_speed'] = train.progress_apply(lambda row: calculate_speed(row.jet_energy, row.jet_mass), axis=1)
test['jet_speed'] = test.progress_apply(lambda row: calculate_speed(row.jet_energy, row.jet_mass), axis=1)

train['speed_x'] = train['jet_speed'] * train['x_div_dist']
test['speed_x'] = test['jet_speed'] * test['x_div_dist']
train['speed_y'] = train['jet_speed'] * train['y_div_dist']
test['speed_y'] = test['jet_speed'] * test['y_div_dist']
train['speed_z'] = train['jet_speed'] * train['z_div_dist']
test['speed_z'] = test['jet_speed'] * test['z_div_dist']

train['speed_xy'] = train['jet_speed'] * train['xy_div_dist']
test['speed_xy'] = test['jet_speed'] * test['xy_div_dist']
train['speed_yz'] = train['jet_speed'] * train['yz_div_dist']
test['speed_yz'] = test['jet_speed'] * test['yz_div_dist']
train['speed_zx'] = train['jet_speed'] * train['zx_div_dist']
test['speed_zx'] = test['jet_speed'] * test['zx_div_dist']

def calculate_travel_time(d, v):
    return np.abs(d) / v if v > 0 else 0

train['time_dis'] = train.progress_apply(lambda row: calculate_travel_time(row.jet_distance, row.jet_speed), axis=1)
test['time_dis'] = test.progress_apply(lambda row: calculate_travel_time(row.jet_distance, row.jet_speed), axis=1)

train['time_x'] = train.progress_apply(lambda row: calculate_travel_time(row.jet_px, row.speed_x), axis=1)
test['time_x'] = test.progress_apply(lambda row: calculate_travel_time(row.jet_px, row.speed_x), axis=1)

train['time_y'] = train.progress_apply(lambda row: calculate_travel_time(row.jet_py, row.speed_y), axis=1)
test['time_y'] = test.progress_apply(lambda row: calculate_travel_time(row.jet_py, row.speed_y), axis=1)

train['time_z'] = train.progress_apply(lambda row: calculate_travel_time(row.jet_pz, row.speed_z), axis=1)
test['time_z'] = test.progress_apply(lambda row: calculate_travel_time(row.jet_pz, row.speed_z), axis=1)

train['time_xy'] = train.progress_apply(lambda row: calculate_travel_time(row.xy_dis, row.speed_xy), axis=1)
test['time_xy'] = test.progress_apply(lambda row: calculate_travel_time(row.xy_dis, row.speed_xy), axis=1)

train['time_yz'] = train.progress_apply(lambda row: calculate_travel_time(row.yz_dis, row.speed_yz), axis=1)
test['time_yz'] = test.progress_apply(lambda row: calculate_travel_time(row.yz_dis, row.speed_yz), axis=1)

train['time_zx'] = train.progress_apply(lambda row: calculate_travel_time(row.zx_dis, row.speed_zx), axis=1)
test['time_zx'] = test.progress_apply(lambda row: calculate_travel_time(row.zx_dis, row.speed_zx), axis=1)

train['jet_mv'] = train['jet_mass'] * train['jet_speed']
test['jet_mv'] = test['jet_mass'] * test['jet_speed']

train['mv_x'] = train['jet_mv'] * train['x_div_dist']
test['mv_x'] = test['jet_mv'] * test['x_div_dist']
train['mv_y'] = train['jet_mv'] * train['y_div_dist']
test['mv_y'] = test['jet_mv'] * test['y_div_dist']
train['mv_z'] = train['jet_mv'] * train['z_div_dist']
test['mv_z'] = test['jet_mv'] * test['z_div_dist']

train['mv_xy'] = train['jet_mv'] * train['xy_div_dist']
test['mv_xy'] = test['jet_mv'] * test['xy_div_dist']
train['mv_yz'] = train['jet_mv'] * train['yz_div_dist']
test['mv_yz'] = test['jet_mv'] * test['yz_div_dist']
train['mv_zx'] = train['jet_mv'] * train['zx_div_dist']
test['mv_zx'] = test['jet_mv'] * test['zx_div_dist']

train['particle_mv'] = train['jet_mv'] / train['number_of_particles_in_this_jet']
test['particle_mv'] = test['jet_mv'] / test['number_of_particles_in_this_jet']

train['particle_mv_x'] = train['particle_mv'] * train['x_div_dist']
test['particle_mv_x'] = test['particle_mv'] * test['x_div_dist']
train['particle_mv_y'] = train['particle_mv'] * train['y_div_dist']
test['particle_mv_y'] = test['particle_mv'] * test['y_div_dist']
train['particle_mv_z'] = train['particle_mv'] * train['z_div_dist']
test['particle_mv_z'] = test['particle_mv'] * test['z_div_dist']

train['particle_mv_xy'] = train['particle_mv'] * train['xy_div_dist']
test['particle_mv_xy'] = test['particle_mv'] * test['xy_div_dist']
train['particle_mv_yz'] = train['particle_mv'] * train['yz_div_dist']
test['particle_mv_yz'] = test['particle_mv'] * test['yz_div_dist']
train['particle_mv_zx'] = train['particle_mv'] * train['zx_div_dist']
test['particle_mv_zx'] = test['particle_mv'] * test['zx_div_dist']

def brute_force(df):
    
    df['event_id_count'] = df.groupby('event_id')['jet_id'].transform('count')
    
    df['event_id_number_particles_max'] = df.groupby('event_id')['number_of_particles_in_this_jet'].transform('max')
    df['event_id_number_particles_mean'] = df.groupby('event_id')['number_of_particles_in_this_jet'].transform('mean')
    df['event_id_number_particles_min'] = df.groupby('event_id')['number_of_particles_in_this_jet'].transform('min')
    df['event_id_number_particles_std'] = df.groupby('event_id')['number_of_particles_in_this_jet'].transform('std')
    
    df['event_id_mass_max'] = df.groupby('event_id')['jet_mass'].transform('max')
    df['event_id_mass_mean'] = df.groupby('event_id')['jet_mass'].transform('mean')
    df['event_id_mass_min'] = df.groupby('event_id')['jet_mass'].transform('min')
    df['event_id_mass_std'] = df.groupby('event_id')['jet_mass'].transform('std')
    
    df['event_id_energy_max'] = df.groupby('event_id')['jet_energy'].transform('max')
    df['event_id_energy_mean'] = df.groupby('event_id')['jet_energy'].transform('mean')
    df['event_id_energy_min'] = df.groupby('event_id')['jet_energy'].transform('min')
    df['event_id_energy_std'] = df.groupby('event_id')['jet_energy'].transform('std')
    
    df['event_id_mass_x_max'] = df.groupby('event_id')['mass_x'].transform('max')
    df['event_id_mass_x_mean'] = df.groupby('event_id')['mass_x'].transform('mean')
    df['event_id_mass_x_min'] = df.groupby('event_id')['mass_x'].transform('min')
    df['event_id_mass_x_std'] = df.groupby('event_id')['mass_x'].transform('std')
    
    df['event_id_mass_y_max'] = df.groupby('event_id')['mass_y'].transform('max')
    df['event_id_mass_y_mean'] = df.groupby('event_id')['mass_y'].transform('mean')
    df['event_id_mass_y_min'] = df.groupby('event_id')['mass_y'].transform('min')
    df['event_id_mass_y_std'] = df.groupby('event_id')['mass_y'].transform('std')
    
    df['event_id_mass_z_max'] = df.groupby('event_id')['mass_z'].transform('max')
    df['event_id_mass_z_mean'] = df.groupby('event_id')['mass_z'].transform('mean')
    df['event_id_mass_z_min'] = df.groupby('event_id')['mass_z'].transform('min')
    df['event_id_mass_z_std'] = df.groupby('event_id')['mass_z'].transform('std')
    
    df['event_id_mass_xy_max'] = df.groupby('event_id')['mass_xy'].transform('max')
    df['event_id_mass_xy_mean'] = df.groupby('event_id')['mass_xy'].transform('mean')
    df['event_id_mass_xy_min'] = df.groupby('event_id')['mass_xy'].transform('min')
    df['event_id_mass_xy_std'] = df.groupby('event_id')['mass_xy'].transform('std')
    
    df['event_id_mass_yz_max'] = df.groupby('event_id')['mass_yz'].transform('max')
    df['event_id_mass_yz_mean'] = df.groupby('event_id')['mass_yz'].transform('mean')
    df['event_id_mass_yz_min'] = df.groupby('event_id')['mass_yz'].transform('min')
    df['event_id_mass_yz_std'] = df.groupby('event_id')['mass_yz'].transform('std')
    
    df['event_id_mass_zx_max'] = df.groupby('event_id')['mass_zx'].transform('max')
    df['event_id_mass_zx_mean'] = df.groupby('event_id')['mass_zx'].transform('mean')
    df['event_id_mass_zx_min'] = df.groupby('event_id')['mass_zx'].transform('min')
    df['event_id_mass_zx_std'] = df.groupby('event_id')['mass_zx'].transform('std')
    
    df['event_id_energy_x_max'] = df.groupby('event_id')['energy_x'].transform('max')
    df['event_id_energy_x_mean'] = df.groupby('event_id')['energy_x'].transform('mean')
    df['event_id_energy_x_min'] = df.groupby('event_id')['energy_x'].transform('min')
    df['event_id_energy_x_std'] = df.groupby('event_id')['energy_x'].transform('std')
    
    df['event_id_energy_y_max'] = df.groupby('event_id')['energy_y'].transform('max')
    df['event_id_energy_y_mean'] = df.groupby('event_id')['energy_y'].transform('mean')
    df['event_id_energy_y_min'] = df.groupby('event_id')['energy_y'].transform('min')
    df['event_id_energy_y_std'] = df.groupby('event_id')['energy_y'].transform('std')
    
    df['event_id_energy_z_max'] = df.groupby('event_id')['energy_z'].transform('max')
    df['event_id_energy_z_mean'] = df.groupby('event_id')['energy_z'].transform('mean')
    df['event_id_energy_z_min'] = df.groupby('event_id')['energy_z'].transform('min')
    df['event_id_energy_z_std'] = df.groupby('event_id')['energy_z'].transform('std')
    
    df['event_id_energy_xy_max'] = df.groupby('event_id')['energy_xy'].transform('max')
    df['event_id_energy_xy_mean'] = df.groupby('event_id')['energy_xy'].transform('mean')
    df['event_id_energy_xy_min'] = df.groupby('event_id')['energy_xy'].transform('min')
    df['event_id_energy_xy_std'] = df.groupby('event_id')['energy_xy'].transform('std')
    
    df['event_id_energy_yz_max'] = df.groupby('event_id')['energy_yz'].transform('max')
    df['event_id_energy_yz_mean'] = df.groupby('event_id')['energy_yz'].transform('mean')
    df['event_id_energy_yz_min'] = df.groupby('event_id')['energy_yz'].transform('min')
    df['event_id_energy_yz_std'] = df.groupby('event_id')['energy_yz'].transform('std')
    
    df['event_id_energy_zx_max'] = df.groupby('event_id')['energy_zx'].transform('max')
    df['event_id_energy_zx_mean'] = df.groupby('event_id')['energy_zx'].transform('mean')
    df['event_id_energy_zx_min'] = df.groupby('event_id')['energy_zx'].transform('min')
    df['event_id_energy_zx_std'] = df.groupby('event_id')['energy_zx'].transform('std')
    
    df['event_id_particles_mass_max'] = df.groupby('event_id')['mean_particles_mass'].transform('max')
    df['event_id_particles_mass_mean'] = df.groupby('event_id')['mean_particles_mass'].transform('mean')
    df['event_id_particles_mass_min'] = df.groupby('event_id')['mean_particles_mass'].transform('min')
    df['event_id_particles_mass_std'] = df.groupby('event_id')['mean_particles_mass'].transform('std')
    
    df['event_id_particles_energy_max'] = df.groupby('event_id')['mean_particles_energy'].transform('max')
    df['event_id_particles_energy_mean'] = df.groupby('event_id')['mean_particles_energy'].transform('mean')
    df['event_id_particles_energy_min'] = df.groupby('event_id')['mean_particles_energy'].transform('min')
    df['event_id_particles_energy_std'] = df.groupby('event_id')['mean_particles_energy'].transform('std')
    
    df['event_id_distance_max'] = df.groupby('event_id')['jet_distance'].transform('max')
    df['event_id_distance_mean'] = df.groupby('event_id')['jet_distance'].transform('mean')
    df['event_id_distance_min'] = df.groupby('event_id')['jet_distance'].transform('min')
    df['event_id_distance_std'] = df.groupby('event_id')['jet_distance'].transform('std')
    
    df['event_id_xy_dis_max'] = df.groupby('event_id')['xy_dis'].transform('max')
    df['event_id_xy_dis_mean'] = df.groupby('event_id')['xy_dis'].transform('mean')
    df['event_id_xy_dis_min'] = df.groupby('event_id')['xy_dis'].transform('min')
    df['event_id_xy_dis_std'] = df.groupby('event_id')['xy_dis'].transform('std')
    
    df['event_id_yz_dis_max'] = df.groupby('event_id')['yz_dis'].transform('max')
    df['event_id_yz_dis_mean'] = df.groupby('event_id')['yz_dis'].transform('mean')
    df['event_id_yz_dis_min'] = df.groupby('event_id')['yz_dis'].transform('min')
    df['event_id_yz_dis_std'] = df.groupby('event_id')['yz_dis'].transform('std')
    
    df['event_id_zx_dis_max'] = df.groupby('event_id')['zx_dis'].transform('max')
    df['event_id_zx_dis_mean'] = df.groupby('event_id')['zx_dis'].transform('mean')
    df['event_id_zx_dis_min'] = df.groupby('event_id')['zx_dis'].transform('min')
    df['event_id_zx_dis_std'] = df.groupby('event_id')['zx_dis'].transform('std')
    
    df['event_id_x_div_dist_max'] = df.groupby('event_id')['x_div_dist'].transform('max')
    df['event_id_x_div_dist_mean'] = df.groupby('event_id')['x_div_dist'].transform('mean')
    df['event_id_x_div_dist_min'] = df.groupby('event_id')['x_div_dist'].transform('min')
    df['event_id_x_div_dist_std'] = df.groupby('event_id')['x_div_dist'].transform('std')
    
    df['event_id_y_div_dist_max'] = df.groupby('event_id')['y_div_dist'].transform('max')
    df['event_id_y_div_dist_mean'] = df.groupby('event_id')['y_div_dist'].transform('mean')
    df['event_id_y_div_dist_min'] = df.groupby('event_id')['y_div_dist'].transform('min')
    df['event_id_y_div_dist_std'] = df.groupby('event_id')['y_div_dist'].transform('std')
    
    df['event_id_z_div_dist_max'] = df.groupby('event_id')['z_div_dist'].transform('max')
    df['event_id_z_div_dist_mean'] = df.groupby('event_id')['z_div_dist'].transform('mean')
    df['event_id_z_div_dist_min'] = df.groupby('event_id')['z_div_dist'].transform('min')
    df['event_id_z_div_dist_std'] = df.groupby('event_id')['z_div_dist'].transform('std')
    
    df['event_id_xy_div_dist_max'] = df.groupby('event_id')['xy_div_dist'].transform('max')
    df['event_id_xy_div_dist_mean'] = df.groupby('event_id')['xy_div_dist'].transform('mean')
    df['event_id_xy_div_dist_min'] = df.groupby('event_id')['xy_div_dist'].transform('min')
    df['event_id_xy_div_dist_std'] = df.groupby('event_id')['xy_div_dist'].transform('std')
    
    df['event_id_yz_div_dist_max'] = df.groupby('event_id')['yz_div_dist'].transform('max')
    df['event_id_yz_div_dist_mean'] = df.groupby('event_id')['yz_div_dist'].transform('mean')
    df['event_id_yz_div_dist_min'] = df.groupby('event_id')['yz_div_dist'].transform('min')
    df['event_id_yz_div_dist_std'] = df.groupby('event_id')['yz_div_dist'].transform('std')
    
    df['event_id_zx_div_dist_max'] = df.groupby('event_id')['zx_div_dist'].transform('max')
    df['event_id_zx_div_dist_mean'] = df.groupby('event_id')['zx_div_dist'].transform('mean')
    df['event_id_zx_div_dist_min'] = df.groupby('event_id')['zx_div_dist'].transform('min')
    df['event_id_zx_div_dist_std'] = df.groupby('event_id')['zx_div_dist'].transform('std')
    
    df['event_id_speed_max'] = df.groupby('event_id')['jet_speed'].transform('max')
    df['event_id_speed_mean'] = df.groupby('event_id')['jet_speed'].transform('mean')
    df['event_id_speed_min'] = df.groupby('event_id')['jet_speed'].transform('min')
    df['event_id_speed_std'] = df.groupby('event_id')['jet_speed'].transform('std')
    
    df['event_id_speed_x_max'] = df.groupby('event_id')['speed_x'].transform('max')
    df['event_id_speed_x_mean'] = df.groupby('event_id')['speed_x'].transform('mean')
    df['event_id_speed_x_min'] = df.groupby('event_id')['speed_x'].transform('min')
    df['event_id_speed_x_std'] = df.groupby('event_id')['speed_x'].transform('std')
    
    df['event_id_speed_y_max'] = df.groupby('event_id')['speed_y'].transform('max')
    df['event_id_speed_y_mean'] = df.groupby('event_id')['speed_y'].transform('mean')
    df['event_id_speed_y_min'] = df.groupby('event_id')['speed_y'].transform('min')
    df['event_id_speed_y_std'] = df.groupby('event_id')['speed_y'].transform('std')
    
    df['event_id_speed_z_max'] = df.groupby('event_id')['speed_z'].transform('max')
    df['event_id_speed_z_mean'] = df.groupby('event_id')['speed_z'].transform('mean')
    df['event_id_speed_z_min'] = df.groupby('event_id')['speed_z'].transform('min')
    df['event_id_speed_z_std'] = df.groupby('event_id')['speed_z'].transform('std')
    
    df['event_id_speed_xy_max'] = df.groupby('event_id')['speed_xy'].transform('max')
    df['event_id_speed_xy_mean'] = df.groupby('event_id')['speed_xy'].transform('mean')
    df['event_id_speed_xy_min'] = df.groupby('event_id')['speed_xy'].transform('min')
    df['event_id_speed_xy_std'] = df.groupby('event_id')['speed_xy'].transform('std')
    
    df['event_id_speed_yz_max'] = df.groupby('event_id')['speed_yz'].transform('max')
    df['event_id_speed_yz_mean'] = df.groupby('event_id')['speed_yz'].transform('mean')
    df['event_id_speed_yz_min'] = df.groupby('event_id')['speed_yz'].transform('min')
    df['event_id_speed_yz_std'] = df.groupby('event_id')['speed_yz'].transform('std')
    
    df['event_id_speed_zx_max'] = df.groupby('event_id')['speed_zx'].transform('max')
    df['event_id_speed_zx_mean'] = df.groupby('event_id')['speed_zx'].transform('mean')
    df['event_id_speed_zx_min'] = df.groupby('event_id')['speed_zx'].transform('min')
    df['event_id_speed_zx_std'] = df.groupby('event_id')['speed_zx'].transform('std')    
    
    df['event_id_px_max'] = df.groupby('event_id')['jet_px'].transform('max')
    df['event_id_px_mean'] = df.groupby('event_id')['jet_px'].transform('mean')
    df['event_id_px_min'] = df.groupby('event_id')['jet_px'].transform('min')
    df['event_id_px_std'] = df.groupby('event_id')['jet_px'].transform('std')
    
    df['event_id_py_max'] = df.groupby('event_id')['jet_py'].transform('max')
    df['event_id_py_mean'] = df.groupby('event_id')['jet_py'].transform('mean')
    df['event_id_py_min'] = df.groupby('event_id')['jet_py'].transform('min')
    df['event_id_py_std'] = df.groupby('event_id')['jet_py'].transform('std')
    
    df['event_id_pz_max'] = df.groupby('event_id')['jet_pz'].transform('max')
    df['event_id_pz_mean'] = df.groupby('event_id')['jet_pz'].transform('mean')
    df['event_id_pz_min'] = df.groupby('event_id')['jet_pz'].transform('min')
    df['event_id_pz_std'] = df.groupby('event_id')['jet_pz'].transform('std')
    
    df['event_id_angle_xy_max'] = df.groupby('event_id')['angle_xy'].transform('max')
    df['event_id_angle_xy_mean'] = df.groupby('event_id')['angle_xy'].transform('mean')
    df['event_id_angle_xy_min'] = df.groupby('event_id')['angle_xy'].transform('min')
    df['event_id_angle_xy_std'] = df.groupby('event_id')['angle_xy'].transform('std')
    
    df['event_id_angle_xz_max'] = df.groupby('event_id')['angle_xz'].transform('max')
    df['event_id_angle_xz_mean'] = df.groupby('event_id')['angle_xz'].transform('mean')
    df['event_id_angle_xz_min'] = df.groupby('event_id')['angle_xz'].transform('min')
    df['event_id_angle_xz_std'] = df.groupby('event_id')['angle_xz'].transform('std')
    
    df['event_id_angle_yx_max'] = df.groupby('event_id')['angle_yx'].transform('max')
    df['event_id_angle_yx_mean'] = df.groupby('event_id')['angle_yx'].transform('mean')
    df['event_id_angle_yx_min'] = df.groupby('event_id')['angle_yx'].transform('min')
    df['event_id_angle_yx_std'] = df.groupby('event_id')['angle_yx'].transform('std')
    
    df['event_id_angle_yz_max'] = df.groupby('event_id')['angle_yz'].transform('max')
    df['event_id_angle_yz_mean'] = df.groupby('event_id')['angle_yz'].transform('mean')
    df['event_id_angle_yz_min'] = df.groupby('event_id')['angle_yz'].transform('min')
    df['event_id_angle_yz_std'] = df.groupby('event_id')['angle_yz'].transform('std')
    
    df['event_id_angle_zy_max'] = df.groupby('event_id')['angle_zy'].transform('max')
    df['event_id_angle_zy_mean'] = df.groupby('event_id')['angle_zy'].transform('mean')
    df['event_id_angle_zy_min'] = df.groupby('event_id')['angle_zy'].transform('min')
    df['event_id_angle_zy_std'] = df.groupby('event_id')['angle_zy'].transform('std')
    
    df['event_id_angle_zx_max'] = df.groupby('event_id')['angle_zx'].transform('max')
    df['event_id_angle_zx_mean'] = df.groupby('event_id')['angle_zx'].transform('mean')
    df['event_id_angle_zx_min'] = df.groupby('event_id')['angle_zx'].transform('min')
    df['event_id_angle_zx_std'] = df.groupby('event_id')['angle_zx'].transform('std')
    
    df['event_id_time_dis_max'] = df.groupby('event_id')['time_dis'].transform('max')
    df['event_id_time_dis_mean'] = df.groupby('event_id')['time_dis'].transform('mean')
    df['event_id_time_dis_min'] = df.groupby('event_id')['time_dis'].transform('min')
    df['event_id_time_dis_std'] = df.groupby('event_id')['time_dis'].transform('std')
    
    df['event_id_time_x_max'] = df.groupby('event_id')['time_x'].transform('max')
    df['event_id_time_x_mean'] = df.groupby('event_id')['time_x'].transform('mean')
    df['event_id_time_x_min'] = df.groupby('event_id')['time_x'].transform('min')
    df['event_id_time_x_std'] = df.groupby('event_id')['time_x'].transform('std')
    
    df['event_id_time_y_max'] = df.groupby('event_id')['time_y'].transform('max')
    df['event_id_time_y_mean'] = df.groupby('event_id')['time_y'].transform('mean')
    df['event_id_time_y_min'] = df.groupby('event_id')['time_y'].transform('min')
    df['event_id_time_y_std'] = df.groupby('event_id')['time_y'].transform('std')
    
    df['event_id_time_z_max'] = df.groupby('event_id')['time_z'].transform('max')
    df['event_id_time_z_mean'] = df.groupby('event_id')['time_z'].transform('mean')
    df['event_id_time_z_min'] = df.groupby('event_id')['time_z'].transform('min')
    df['event_id_time_z_std'] = df.groupby('event_id')['time_z'].transform('std')
    
    df['event_id_time_xy_max'] = df.groupby('event_id')['time_xy'].transform('max')
    df['event_id_time_xy_mean'] = df.groupby('event_id')['time_xy'].transform('mean')
    df['event_id_time_xy_min'] = df.groupby('event_id')['time_xy'].transform('min')
    df['event_id_time_xy_std'] = df.groupby('event_id')['time_xy'].transform('std')
    
    df['event_id_time_yz_max'] = df.groupby('event_id')['time_yz'].transform('max')
    df['event_id_time_yz_mean'] = df.groupby('event_id')['time_yz'].transform('mean')
    df['event_id_time_yz_min'] = df.groupby('event_id')['time_yz'].transform('min')
    df['event_id_time_yz_std'] = df.groupby('event_id')['time_yz'].transform('std')
    
    df['event_id_time_zx_max'] = df.groupby('event_id')['time_zx'].transform('max')
    df['event_id_time_zx_mean'] = df.groupby('event_id')['time_zx'].transform('mean')
    df['event_id_time_zx_min'] = df.groupby('event_id')['time_zx'].transform('min')
    df['event_id_time_zx_std'] = df.groupby('event_id')['time_zx'].transform('std')
    
    df['event_id_mv_max'] = df.groupby('event_id')['jet_mv'].transform('max')
    df['event_id_mv_mean'] = df.groupby('event_id')['jet_mv'].transform('mean')
    df['event_id_mv_min'] = df.groupby('event_id')['jet_mv'].transform('min')
    df['event_id_mv_std'] = df.groupby('event_id')['jet_mv'].transform('std')
    
    df['event_id_mv_x_max'] = df.groupby('event_id')['mv_x'].transform('max')
    df['event_id_mv_x_mean'] = df.groupby('event_id')['mv_x'].transform('mean')
    df['event_id_mv_x_min'] = df.groupby('event_id')['mv_x'].transform('min')
    df['event_id_mv_x_std'] = df.groupby('event_id')['mv_x'].transform('std')
    
    df['event_id_mv_y_max'] = df.groupby('event_id')['mv_y'].transform('max')
    df['event_id_mv_y_mean'] = df.groupby('event_id')['mv_y'].transform('mean')
    df['event_id_mv_y_min'] = df.groupby('event_id')['mv_y'].transform('min')
    df['event_id_mv_y_std'] = df.groupby('event_id')['mv_y'].transform('std')
    
    df['event_id_mv_z_max'] = df.groupby('event_id')['mv_z'].transform('max')
    df['event_id_mv_z_mean'] = df.groupby('event_id')['mv_z'].transform('mean')
    df['event_id_mv_z_min'] = df.groupby('event_id')['mv_z'].transform('min')
    df['event_id_mv_z_std'] = df.groupby('event_id')['mv_z'].transform('std')
    
    df['event_id_mv_xy_max'] = df.groupby('event_id')['mv_xy'].transform('max')
    df['event_id_mv_xy_mean'] = df.groupby('event_id')['mv_xy'].transform('mean')
    df['event_id_mv_xy_min'] = df.groupby('event_id')['mv_xy'].transform('min')
    df['event_id_mv_xy_std'] = df.groupby('event_id')['mv_xy'].transform('std')
    
    df['event_id_mv_yz_max'] = df.groupby('event_id')['mv_yz'].transform('max')
    df['event_id_mv_yz_mean'] = df.groupby('event_id')['mv_yz'].transform('mean')
    df['event_id_mv_yz_min'] = df.groupby('event_id')['mv_yz'].transform('min')
    df['event_id_mv_yz_std'] = df.groupby('event_id')['mv_yz'].transform('std')
    
    df['event_id_mv_zx_max'] = df.groupby('event_id')['mv_zx'].transform('max')
    df['event_id_mv_zx_mean'] = df.groupby('event_id')['mv_zx'].transform('mean')
    df['event_id_mv_zx_min'] = df.groupby('event_id')['mv_zx'].transform('min')
    df['event_id_mv_zx_std'] = df.groupby('event_id')['mv_zx'].transform('std')
    
    df['event_id_particle_mv_max'] = df.groupby('event_id')['particle_mv'].transform('max')
    df['event_id_particle_mv_mean'] = df.groupby('event_id')['particle_mv'].transform('mean')
    df['event_id_particle_mv_min'] = df.groupby('event_id')['particle_mv'].transform('min')
    df['event_id_particle_mv_std'] = df.groupby('event_id')['particle_mv'].transform('std')
    
    df['event_id_particle_mv_x_max'] = df.groupby('event_id')['particle_mv_x'].transform('max')
    df['event_id_particle_mv_x_mean'] = df.groupby('event_id')['particle_mv_x'].transform('mean')
    df['event_id_particle_mv_x_min'] = df.groupby('event_id')['particle_mv_x'].transform('min')
    df['event_id_particle_mv_x_std'] = df.groupby('event_id')['particle_mv_x'].transform('std')
    
    df['event_id_particle_mv_y_max'] = df.groupby('event_id')['particle_mv_y'].transform('max')
    df['event_id_particle_mv_y_mean'] = df.groupby('event_id')['particle_mv_y'].transform('mean')
    df['event_id_particle_mv_y_min'] = df.groupby('event_id')['particle_mv_y'].transform('min')
    df['event_id_particle_mv_y_std'] = df.groupby('event_id')['particle_mv_y'].transform('std')
    
    df['event_id_particle_mv_z_max'] = df.groupby('event_id')['particle_mv_z'].transform('max')
    df['event_id_particle_mv_z_mean'] = df.groupby('event_id')['particle_mv_z'].transform('mean')
    df['event_id_particle_mv_z_min'] = df.groupby('event_id')['particle_mv_z'].transform('min')
    df['event_id_particle_mv_z_std'] = df.groupby('event_id')['particle_mv_z'].transform('std')
    
    df['event_id_particle_mv_xy_max'] = df.groupby('event_id')['particle_mv_xy'].transform('max')
    df['event_id_particle_mv_xy_mean'] = df.groupby('event_id')['particle_mv_xy'].transform('mean')
    df['event_id_particle_mv_xy_min'] = df.groupby('event_id')['particle_mv_xy'].transform('min')
    df['event_id_particle_mv_xy_std'] = df.groupby('event_id')['particle_mv_xy'].transform('std')
    
    df['event_id_particle_mv_yz_max'] = df.groupby('event_id')['particle_mv_yz'].transform('max')
    df['event_id_particle_mv_yz_mean'] = df.groupby('event_id')['particle_mv_yz'].transform('mean')
    df['event_id_particle_mv_yz_min'] = df.groupby('event_id')['particle_mv_yz'].transform('min')
    df['event_id_particle_mv_yz_std'] = df.groupby('event_id')['particle_mv_yz'].transform('std')
    
    df['event_id_particle_mv_zx_max'] = df.groupby('event_id')['particle_mv_zx'].transform('max')
    df['event_id_particle_mv_zx_mean'] = df.groupby('event_id')['particle_mv_zx'].transform('mean')
    df['event_id_particle_mv_zx_min'] = df.groupby('event_id')['particle_mv_zx'].transform('min')
    df['event_id_particle_mv_zx_std'] = df.groupby('event_id')['particle_mv_zx'].transform('std')

    df['event_id_particle_time_x_max'] = df.groupby('event_id')['particle_time_x'].transform('max')
    df['event_id_particle_time_x_mean'] = df.groupby('event_id')['particle_time_x'].transform('mean')
    df['event_id_particle_time_x_min'] = df.groupby('event_id')['particle_time_x'].transform('min')
    df['event_id_particle_time_x_std'] = df.groupby('event_id')['particle_time_x'].transform('std')
    
    df['event_id_particle_time_y_max'] = df.groupby('event_id')['particle_time_y'].transform('max')
    df['event_id_particle_time_y_mean'] = df.groupby('event_id')['particle_time_y'].transform('mean')
    df['event_id_particle_time_y_min'] = df.groupby('event_id')['particle_time_y'].transform('min')
    df['event_id_particle_time_y_std'] = df.groupby('event_id')['particle_time_y'].transform('std')
    
    df['event_id_particle_time_z_max'] = df.groupby('event_id')['particle_time_z'].transform('max')
    df['event_id_particle_time_z_mean'] = df.groupby('event_id')['particle_time_z'].transform('mean')
    df['event_id_particle_time_z_min'] = df.groupby('event_id')['particle_time_z'].transform('min')
    df['event_id_particle_time_z_std'] = df.groupby('event_id')['particle_time_z'].transform('std')
    
    df['event_id_particle_time_xy_max'] = df.groupby('event_id')['particle_time_xy'].transform('max')
    df['event_id_particle_time_xy_mean'] = df.groupby('event_id')['particle_time_xy'].transform('mean')
    df['event_id_particle_time_xy_min'] = df.groupby('event_id')['particle_time_xy'].transform('min')
    df['event_id_particle_time_xy_std'] = df.groupby('event_id')['particle_time_xy'].transform('std')
    
    df['event_id_particle_time_yz_max'] = df.groupby('event_id')['particle_time_yz'].transform('max')
    df['event_id_particle_time_yz_mean'] = df.groupby('event_id')['particle_time_yz'].transform('mean')
    df['event_id_particle_time_yz_min'] = df.groupby('event_id')['particle_time_yz'].transform('min')
    df['event_id_particle_time_yz_std'] = df.groupby('event_id')['particle_time_yz'].transform('std')
    
    df['event_id_particle_time_zx_max'] = df.groupby('event_id')['particle_time_zx'].transform('max')
    df['event_id_particle_time_zx_mean'] = df.groupby('event_id')['particle_time_zx'].transform('mean')
    df['event_id_particle_time_zx_min'] = df.groupby('event_id')['particle_time_zx'].transform('min')
    df['event_id_particle_time_zx_std'] = df.groupby('event_id')['particle_time_zx'].transform('std')
     
    return df

train = brute_force(train)
test = brute_force(test)

train['event_id_energy_sum'] = train.groupby('event_id')['jet_energy'].transform('sum')
test['event_id_energy_sum'] = test.groupby('event_id')['jet_energy'].transform('sum')
train['event_id_mass_sum'] = train.groupby('event_id')['jet_mass'].transform('sum')
test['event_id_mass_sum'] = test.groupby('event_id')['jet_mass'].transform('sum')
train['event_id_speed'] = train.progress_apply(lambda row: calculate_speed(row.event_id_energy_sum, row.event_id_mass_sum), axis=1)
test['event_id_speed'] = test.progress_apply(lambda row: calculate_speed(row.event_id_energy_sum, row.event_id_mass_sum), axis=1)


train.info()
test.info()

train.to_pickle('train.pickle')
test.to_pickle('test.pickle')

# PART 3: prepare target and features

mapping_dict = {21:0, 1:1, 4:2, 5:3}
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

train['label'] = train.label.map(mapping_dict)

use_features = [col for col in train.columns if col not in ['jet_id', 'event_id', 'label']]

# PART 4: model and cross validation training

def run_lgb(df_train, df_test, use_features):
    
    target = 'label'
    oof_pred = np.zeros((len(df_train), 4))
    y_pred = np.zeros((len(df_test), 4))
    
    folds = GroupKFold(n_splits=5)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['label'], train['event_id'])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        
        params = {
            'learning_rate': 0.1,
            'metric': 'multiclass',
            'objective': 'multiclass',
            'num_classes': 4,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 2,
            'n_jobs': -1,
            'seed': 1029,
            'max_depth': 10,
            'num_leaves': 100,
            'lambda_l1': 0.5,
            'lambda_l2': 0.8
        }
        
        model = lgb.train(params, 
                          train_set, 
                          num_boost_round=500,
                          early_stopping_rounds=100,
                          valid_sets=[train_set, val_set],
                          verbose_eval=100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features]) / folds.n_splits
        
        y_one_hot = label_binarize(y_val, np.arange(4)) 
        oof_one_hot = label_binarize(oof_pred[val_ind].argmax(axis=1), np.arange(4)) 
        score = roc_auc_score(y_one_hot, oof_one_hot) 
        print('auc: ', score)
        
        print("Features importance...")
        gain = model.feature_importance('gain')
        feat_imp = pd.DataFrame({'feature': model.feature_name(), 
                         'split': model.feature_importance('split'), 
                         'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        print('Top 50 features:\n', feat_imp.head(50))
        
        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()
        
    return y_pred, oof_pred


y_pred, oof_pred = run_lgb(train, test, use_features)

y_one_hot = label_binarize(train['label'], np.arange(4)) 
oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
score = roc_auc_score(y_one_hot, oof_one_hot) 
print('auc: ', score)

# PART 5: predict and save submission

submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
submission.label = y_pred.argmax(axis=1)
submission.label = submission.label.map(mapping_dict_inv)
submission.head()

submission.to_csv('submission_lgb_baseline.csv', index=False)