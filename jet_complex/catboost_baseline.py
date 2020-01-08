__author__ = 'zhengheng'

#################################### 第零部分 ####################################
## 导入相关库

import warnings
warnings.simplefilter('ignore')

import re
import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.6f' % x)

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from catboost import CatBoostClassifier

#################################### 第一部分 ####################################
## particle 特征工程及预处理

# 导入数据

train = pd.read_csv('./jet_complex_data/complex_train_R04_particle.csv') 
test = pd.read_csv('./jet_complex_data/complex_test_R04_particle.csv')

df = pd.concat([train, test], axis=0)

train_length = len(train)

del train, test
gc.collect()


# "三维距离"特征, 生成大概需要 4 分钟
# 使用 np.vectorize 进行加速, 原生 dataframe apply 函数实在太慢了
def l2_dict(x, y, z): 
    return np.sqrt(np.square(x)+np.square(y)+np.square(z))

do_in_vec = np.vectorize(l2_dict, otypes=[np.float])
vec = do_in_vec(df.particle_px, df.particle_py, df.particle_pz)
df['particle_distance'] = vec

# x 方向在 "三维距离" 上的映射距离
df['x_div_dist'] = df['particle_px'] / df['particle_distance']

# energy 在 x 方向上的映射
df['energy_x'] = df['particle_energy'] * df['x_div_dist']

# mass 在 x 方向上的映射
df['mass_x'] = df['particle_mass'] * df['x_div_dist']

# x 与 y/z 的夹角
def angle(x, y):
    return np.degrees(np.math.atan(x / y)) if y != 0 else 0

do_in_vec = np.vectorize(angle, otypes=[np.float])

vec = do_in_vec(df.particle_px, df.particle_py)
df['angle_xy'] = vec

vec = do_in_vec(df.particle_px, df.particle_pz)
df['angle_xz'] = vec

# 速度, 根据 E = (1/2)mv^2 来推算
def calculate_speed(e, m):
    return np.sqrt(2*e/m) if m > 0 else 0

do_in_vec = np.vectorize(calculate_speed, otypes=[np.float])
vec = do_in_vec(df.particle_energy, df.particle_mass)
df['particle_speed'] = vec

# speed 在 x 方向上的映射
df['speed_x'] = df['particle_speed'] * df['x_div_dist']

# time 在 x 方向上的映射
# 有速度有"距离"就可以算时间了, 需要注意的是 particle_px 并不是真正的距离, 
# 其实是之前误解了这个特征的含义, 但好像还有用, 就留下来了
def calculate_travel_time(d, v):
    return np.abs(d) / v if v > 0 else 0

do_in_vec = np.vectorize(calculate_travel_time, otypes=[np.float])

vec = do_in_vec(df.particle_px, df.speed_x)
df['time_x'] = vec

# 生成基于 jet_id 的统计特征
def brute_force(df):
    # 从之前 jet_simple 的结果来看, std 是最有用的统计方式, 在内存不足的情况下, 先只用 std
    df['jet_id_mass_std'] = df.groupby('jet_id')['particle_mass'].transform('std')
    df['jet_id_energy_std'] = df.groupby('jet_id')['particle_energy'].transform('std')
    df['jet_id_mass_x_std'] = df.groupby('jet_id')['mass_x'].transform('std')
    df['jet_id_energy_x_std'] = df.groupby('jet_id')['energy_x'].transform('std')
    df['jet_id_x_div_dist_std'] = df.groupby('jet_id')['x_div_dist'].transform('std')
    df['jet_id_speed_std'] = df.groupby('jet_id')['particle_speed'].transform('std')
    df['jet_id_speed_x_std'] = df.groupby('jet_id')['speed_x'].transform('std')
    df['jet_id_px_std'] = df.groupby('jet_id')['particle_px'].transform('std')
    df['jet_id_angle_xy_std'] = df.groupby('jet_id')['angle_xy'].transform('std')
    df['jet_id_angle_xz_std'] = df.groupby('jet_id')['angle_xz'].transform('std')
    df['jet_id_time_x_std'] = df.groupby('jet_id')['time_x'].transform('std')

    # particle 类别特征, 猜测互为相反数的是正负电子, 没有相反数的是中性的中子/光子 
    df['particle_category_abs'] = np.abs(df['particle_category'])
    df['particle_category_unique_len'] = df.groupby(['jet_id'])['particle_category'].transform('unique').apply(len)
    df['particle_category_unique_len_abs'] = df.groupby(['jet_id'])['particle_category_abs'].transform('unique').apply(len)
    
    return df

df = brute_force(df)

# energy 和 mass 基于 jet_id 计算总和, 后续可以结合 jet 数据来生成是否有能量/质量损失(TODO)
df['jet_id_energy_sum'] = df.groupby('jet_id')['particle_energy'].transform('sum')
df['jet_id_mass_sum'] = df.groupby('jet_id')['particle_mass'].transform('sum')

# 调整下数据类型, 降低内存使用
df['particle_category'] = df['particle_category'].astype('int16')
df['particle_category_abs'] = df['particle_category_abs'].astype('int16')
df['particle_category_unique_len'] = df['particle_category_unique_len'].astype('int16')
df['particle_category_unique_len_abs'] = df['particle_category_unique_len_abs'].astype('int16')

for col in tqdm([i for i in df.columns.tolist() if i not in ['particle_category', 'particle_category_abs', 'particle_category_unique_len', 'particle_category_unique_len_abs', 'jet_id']]):
    df[col] = df[col].astype('float16')

# 调整下 columns 顺序, 方便后续处理
df = df.reindex(columns=[i for i in df.columns.tolist() if i != 'jet_id'] + ['jet_id'])

print("particle feature engineering done.")
print(df.info())

print("saving...")
train = df[:train_length]
test = df[train_length:]
train.to_pickle('train_particle_32cols.pickle')
test.to_pickle('test_particle_32cols.pickle')

del train, test, df
gc.collect()

#################################### 第二部分 ####################################
## jet 特征工程及预处理

# 这部分的处理基本与 particle 基本一致, 沿用了 jet_simple 里面的特征生成方式, 比较暴力, 不多解释

train = pd.read_csv('./jet_complex_data/complex_train_R04_jet.csv')
test = pd.read_csv('./jet_complex_data/complex_test_R04_jet.csv')

df = pd.concat([train, test], axis=0)

train_length = len(train)

del train, test
gc.collect()

def l2_dict(x, y, z): 
    return np.sqrt(np.square(x)+np.square(y)+np.square(z))

df['jet_distance'] = df.progress_apply(lambda row: l2_dict(row.jet_px, row.jet_py, row.jet_pz), axis=1)
df['xy_dis'] = df.progress_apply(lambda row: l2_dict(row.jet_px, row.jet_py, 0), axis=1)
df['yz_dis'] = df.progress_apply(lambda row: l2_dict(0, row.jet_py, row.jet_pz), axis=1)
df['zx_dis'] = df.progress_apply(lambda row: l2_dict(row.jet_px, 0, row.jet_pz), axis=1)

df['x_div_dist'] = df['jet_px'] / df['jet_distance']
df['y_div_dist'] = df['jet_py'] / df['jet_distance']
df['z_div_dist'] = df['jet_pz'] / df['jet_distance']
df['xy_div_dist'] = df['xy_dis'] / df['jet_distance']
df['yz_div_dist'] = df['yz_dis'] / df['jet_distance']
df['zx_div_dist'] = df['zx_dis'] / df['jet_distance']

df['energy_x'] = df['jet_energy'] * df['x_div_dist']
df['energy_y'] = df['jet_energy'] * df['y_div_dist']
df['energy_z'] = df['jet_energy'] * df['z_div_dist']
df['energy_xy'] = df['jet_energy'] * df['xy_div_dist']
df['energy_yz'] = df['jet_energy'] * df['yz_div_dist']
df['energy_zx'] = df['jet_energy'] * df['zx_div_dist']

def fill_zero(x):
    return 0 if x < 0 else x

df.jet_mass = df.jet_mass.progress_apply(lambda x: fill_zero(x))

df['mass_x'] = df['jet_mass'] * df['x_div_dist']
df['mass_y'] = df['jet_mass'] * df['y_div_dist']
df['mass_z'] = df['jet_mass'] * df['z_div_dist']
df['mass_xy'] = df['jet_mass'] * df['xy_div_dist']
df['mass_yz'] = df['jet_mass'] * df['yz_div_dist']
df['mass_zx'] = df['jet_mass'] * df['zx_div_dist']

def angle(x, y):
    return np.degrees(np.math.atan(x / y)) if y != 0 else 0

df['angle_xy'] = df.progress_apply(lambda row: angle(row.jet_px, row.jet_py), axis=1)
df['angle_yx'] = df.progress_apply(lambda row: angle(row.jet_py, row.jet_px), axis=1)
df['angle_yz'] = df.progress_apply(lambda row: angle(row.jet_py, row.jet_pz), axis=1)
df['angle_zy'] = df.progress_apply(lambda row: angle(row.jet_pz, row.jet_py), axis=1)
df['angle_zx'] = df.progress_apply(lambda row: angle(row.jet_pz, row.jet_px), axis=1)
df['angle_xz'] = df.progress_apply(lambda row: angle(row.jet_px, row.jet_pz), axis=1)

df['mean_particles_mass'] = df['jet_mass'] / df['number_of_particles_in_this_jet']
df['mean_particles_energy'] = df['jet_energy'] / df['number_of_particles_in_this_jet']

def calculate_speed(e, m):
    return np.sqrt(2*e/m) if m > 0 else 0

df['jet_speed'] = df.progress_apply(lambda row: calculate_speed(row.jet_energy, row.jet_mass), axis=1)

df['speed_x'] = df['jet_speed'] * df['x_div_dist']
df['speed_y'] = df['jet_speed'] * df['y_div_dist']
df['speed_z'] = df['jet_speed'] * df['z_div_dist']
df['speed_xy'] = df['jet_speed'] * df['xy_div_dist']
df['speed_yz'] = df['jet_speed'] * df['yz_div_dist']
df['speed_zx'] = df['jet_speed'] * df['zx_div_dist']

def calculate_travel_time(d, v):
    return np.abs(d) / v if v > 0 else 0

df['time_dis'] = df.progress_apply(lambda row: calculate_travel_time(row.jet_distance, row.jet_speed), axis=1)
df['time_x'] = df.progress_apply(lambda row: calculate_travel_time(row.jet_px, row.speed_x), axis=1)
df['time_y'] = df.progress_apply(lambda row: calculate_travel_time(row.jet_py, row.speed_y), axis=1)
df['time_z'] = df.progress_apply(lambda row: calculate_travel_time(row.jet_pz, row.speed_z), axis=1)
df['time_xy'] = df.progress_apply(lambda row: calculate_travel_time(row.xy_dis, row.speed_xy), axis=1)
df['time_yz'] = df.progress_apply(lambda row: calculate_travel_time(row.yz_dis, row.speed_yz), axis=1)
df['time_zx'] = df.progress_apply(lambda row: calculate_travel_time(row.zx_dis, row.speed_zx), axis=1)

df['jet_mv'] = df['jet_mass'] * df['jet_speed']
df['mv_x'] = df['jet_mv'] * df['x_div_dist']
df['mv_y'] = df['jet_mv'] * df['y_div_dist']
df['mv_z'] = df['jet_mv'] * df['z_div_dist']
df['mv_xy'] = df['jet_mv'] * df['xy_div_dist']
df['mv_yz'] = df['jet_mv'] * df['yz_div_dist']
df['mv_zx'] = df['jet_mv'] * df['zx_div_dist']

df['particle_mv'] = df['jet_mv'] / df['number_of_particles_in_this_jet']
df['particle_mv_x'] = df['particle_mv'] * df['x_div_dist']
df['particle_mv_y'] = df['particle_mv'] * df['y_div_dist']
df['particle_mv_z'] = df['particle_mv'] * df['z_div_dist']
df['particle_mv_xy'] = df['particle_mv'] * df['xy_div_dist']
df['particle_mv_yz'] = df['particle_mv'] * df['yz_div_dist']
df['particle_mv_zx'] = df['particle_mv'] * df['zx_div_dist']

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
    
    return df

df = brute_force(df)

df['event_id_energy_sum'] = df.groupby('event_id')['jet_energy'].transform('sum')
df['event_id_mass_sum'] = df.groupby('event_id')['jet_mass'].transform('sum')
df['event_id_speed'] = df.progress_apply(lambda row: calculate_speed(row.event_id_energy_sum, row.event_id_mass_sum), axis=1)

# 处理下数据类型, 降低内存使用
for col in tqdm(df.columns.tolist()):
    if df[col].dtypes == 'float64':
        df[col] = df[col].astype('float16')
    if df[col].dtypes == 'int64':
        df[col] = df[col].astype('int16')

print("jet feature engineering done.")
print(df.info())

print("saving...")
train = df[:train_length]
test = df[train_length:]
train.to_pickle('train_jet.float16.pickle')
test.to_pickle('test_jet.float16.pickle')

del train, test, df
gc.collect())


#################################### 第三部分 ####################################
## 合并 particle 和 jet 特征
# 合并思路

# 数据含有三部分，有从属关系，从上至下分为三层：
# 1. event
# 2. jet
# 3. particle

# 因为同一个 event 的 target 是一样的，因此可以尝试将三层数据合并为一层，用整个 event 来当做输入，进行训练和预测：
# 1. 将 20 个 particle 合并为一个 jet，合并的顺序为按 abs(particle_px) 从大到小排序，如果超过 20 则截断，小于 20 则补零；
# 2. 将上一步的 particle data 按 jet_id 与 jet 数据合并；
# 3. 将 4 个 jet_id 合并为一个 event，合并的顺序为按 abs(jet_px) 从大到小排序，如果超过 4 则截断，小于 4 则补零；
# 4. 将上一步的 jet data 按 event_id 与 event 数据合并。

# PS: 超级耗内存, 如果有 100G 内存可以跑, 不然就拆开 train/test 分别跑吧

train = pd.read_pickle('train_particle_32cols.pickle') 
test = pd.read_pickle('test_particle_32cols.pickle')

df = pd.concat([train, test], axis=0)

train_length = len(train)

del train, test
gc.collect()

n_particles = 20
n_jets = 4

# 按 abs(particle_px) 从大到小将每个 jet_id 里面的 particle 数据进行排序
df['particle_px_abs'] = np.abs(df['particle_px'])
df = df.sort_index(by=['jet_id', 'particle_px_abs'], ascending=False).reset_index()
df = df.drop(['particle_px_abs'], axis=1)

# 生成一些辅助变量
# di 记录的是 {jet_id: [当前 jet_id 已加入的 particle 个数, 当前 jet_id 的序号]}
jet = df.jet_id.unique()
num_total_jet = jet.shape[0]

di = dict()
for i, j in tqdm(enumerate(jet.tolist())):
    di[j] = [0, i]

# 这两项需要手动整理, 有点蛋疼
common_cols = ['jet_id_mass_std', 'jet_id_mass_x_std', 'jet_id_energy_x_std', 
    'jet_id_x_div_dist_std', 'jet_id_speed_std', 'jet_id_speed_x_std', 'jet_id_px_std',
    'jet_id_angle_xy_std', 'jet_id_angle_xz_std', 'jet_id_time_x_std',
    'particle_category_unique_len', 'particle_category_unique_len_abs', 
    'jet_id_energy_sum', 'jet_id_mass_sum']

particle_cols = ['particle_category', 'particle_category_abs', 'particle_px', 'particle_py', 'particle_pz',
    'particle_energy', 'particle_mass', 'x_div_dist', 'particle_distance', 'particle_speed',
    'speed_x', 'energy_x', 'mass_x', 'time_x', 'angle_xy', 'angle_xz']

useless_cols = ['index', 'jet_id']

col_shape = n_particles * len(particle_cols) + len(common_cols)

df = df.reindex(columns=common_cols+particle_cols+useless_cols)

# 用 np.ndarry 来做处理，使用 dataframe 会相当慢

particle_data = np.zeros(shape=(num_total_jet, col_shape), dtype=np.float16)
len_commons = len(common_cols)
len_particles = len(particle_cols)

for row in tqdm(df.values):
    # 取出该行 particle 的 jet_id
    jet_id = row[-1]
    # n: 当前 jet_id 已加入的 particle 个数
    # i: 当前 jet_id 的序号
    n, i = di[jet_id]
    # 如果当前 jet_id 的 particle 个数少于 n_particles 则加入进来
    if n < n_particles:
        # 首先是 common cols, 只在第一次时添加
        if n == 0:
            particle_data[i, 0:len_commons+len_particles] = row[0:len_commons+len_particles]
        else:
            particle_data[i, len_commons+n*len_particles:len_commons+(n+1)*len_particles] = row[len_commons:len_commons+len_particles]
        # particle 个数加 1
        di[jet_id][0] += 1

del df
gc.collect()

# 合并完的 particle 数据
df_particle = pd.DataFrame(particle_data)
df_particle.columns = common_cols + [f'p{i}_{j}' for i in range(n_particles) for j in particle_cols]
df_particle['jet_id'] = jet

df_particle['particle_category_unique_len'] = df_particle['particle_category_unique_len'].astype('int')
df_particle['particle_category_unique_len_abs'] = df_particle['particle_category_unique_len_abs'].astype('int')

for i in tqdm(range(n_particles)):
    df_particle[f'p{i}_particle_category'] = df_particle[f'p{i}_particle_category'].astype('int')
    df_particle[f'p{i}_particle_category_abs'] = df_particle[f'p{i}_particle_category_abs'].astype('int')

# 读取之前保存的 jet 特征, 下面进行合并
train_jet = pd.read_pickle('train_jet.float16.pickle')
test_jet = pd.read_pickle('test_jet.float16.pickle')

df_jet = pd.concat([train_jet, test_jet], axis=0)

del train_jet, test_jet
gc.collect()

# particle 数据与 jet 数据合并
df = pd.merge(df_jet, df_particle, on=['jet_id'])

del df_particle, df_jet, particle_data
gc.collect()

for col in (df.columns.tolist()):
    if df[col].dtypes == 'float64':
        df[col] = df[col].astype('float16')

# 按 abs(jet_px) 从大到小将每个 event_id 里面的 jet 数据进行排序
df['jet_px_abs'] = np.abs(df['jet_px'])
df = df.sort_index(by=['event_id', 'jet_px_abs'], ascending=False).reset_index()
df = df.drop(['jet_px_abs'], axis=1)

# 需要合并的特征
cols = [col for col in df.columns.tolist() if col not in ['index', 'jet_id', 'event_id']]
# 整理下 columns 次序
df = df.reindex(columns=cols+['index', 'jet_id', 'event_id'])

# 生成一些辅助变量
# di 记录的是 {event_id: [当前 event_id 已加入的 jet 个数, 当前 event_id 的序号]}
event = df.event_id.unique()
num_total_event = event.shape[0]

di = dict()
for i, j in tqdm(enumerate(event.tolist())):
    di[j] = [0, i]

# common_cols 是 基于 event_id 的统计特征, 每个 jet 都一样的
common_cols = [col for col in df.columns.tolist() if col.startswith('event_id_')]
useless_cols = ['index', 'jet_id', 'event_id']
# jet_cols 是每个 jet 不同的特征
jet_cols = [col for col in df.columns.tolist() if col not in common_cols+useless_cols]

col_shape = n_jets * len(jet_cols) + len(common_cols)

df = df.reindex(columns=common_cols+jet_cols+useless_cols)

gc.collect()

# 用 np.ndarry 来做处理，使用 dataframe 会相当慢

jet_data = np.zeros(shape=(num_total_event, col_shape), dtype=np.float16)
len_commons = len(common_cols)
len_jets = len(jet_cols)

for row in tqdm(df.values):
    # 取出该行 jet 的 event_id
    event_id = row[-1]
    # n: 当前 event_id 已加入的 jet 个数
    # i: 当前 event_id 的序号
    n, i = di[event_id]
    # 如果当前 event_id 的 jet 个数少于 4 则加入进来
    if n < n_jets:
        # 首先是 common cols, 只在第一次时添加
        if n == 0:
            jet_data[i, 0:len_commons+len_jets] = row[0:len_commons+len_jets]
        else:
            jet_data[i, len_commons+n*len_jets:len_commons+(n+1)*len_jets] = row[len_commons:len_commons+len_jets]
        # particle 个数加 1
        di[event_id][0] += 1

del df
gc.collect()

# 合并完的 jet 数据
df_jet = pd.DataFrame(jet_data)

df_jet.columns = common_cols + [f'j{i}_{j}' for i in range(n_jets) for j in jet_cols]
df_jet['event_id'] = event

del jet_data
gc.collect()

# 最后合并 event 数据 (其实只有一个特征)
train_event = pd.read_csv('jet_complex_data/complex_train_R04_event.csv')
test_event = pd.read_csv('jet_complex_data/complex_test_R04_event.csv')
df_event = pd.concat([train_event, test_event], axis=0)

del train_event, test_event
gc.collect()

df = pd.merge(df_jet, df_event, on=['event_id'])

gc.collect()

# 标签保留一个就行了
df['label'] = df['j0_label']
df = df.drop([f'j{i}_label' for i in range(n_jets)], axis=1)

# 再次调整 columns 次序, 在 nn 训练时容易分割特征
cols = [col for col in df.columns.tolist() if col not in ['number_of_jet_in_this_event', 'event_id', 'label']]
df = df.reindex(columns=['number_of_jet_in_this_event']+cols+['event_id', 'label'])

# 至此, 特征分成了三个部分
# 1. event 特征
# 2. jet 特征
# 3. particle 特征
c_cols = ['number_of_jet_in_this_event'] + [col for col in df.columns.tolist() if col.startswith('event_id_')]
p_cols = [col for col in df.columns.tolist() if re.match(r'j\d+_p\d+', col)]
j_cols = [col for col in df.columns.tolist() if col not in p_cols+c_cols+['event_id', 'label']]
df = df.reindex(columns=c_cols+j_cols+p_cols+['event_id', 'label'])

print("merge done")
print(df.info())

print("saving...")
train = df[:train_length]
test = df[train_length:]
train.to_pickle('train_04j20p_fe_v3.pickle')
test.to_pickle('test_04j20p_fe_v3.pickle')

del df, df_jet, df_event, train, test
gc.collect()

#################################### 第四部分 ####################################
# CatBoost CV training

train = pd.read_pickle('train_04j20p_fe_v3.pickle')
test = pd.read_pickle('test_04j20p_fe_v3.pickle')

mapping_dict = {21:0, 1:1, 4:2, 5:3}
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

train['label'] = train['label'].astype('int')
train['label'] = train.label.map(mapping_dict)

use_features = [col for col in train.columns if col not in ['event_id', 'label']]

def run_cat(df_train, df_test, use_features):
    
    target = 'label'
    oof_pred = np.zeros((len(df_train), 4))
    y_pred = np.zeros((len(df_test), 4))
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['label'])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        
        params = {
            'task_type': 'GPU',
            'learning_rate': 0.1,
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
            'classes_count': 4,
            'iterations': 10000,
            'random_seed': 1029,
            'max_depth': 8,
            'max_leaves': 64,
            'reg_lambda': 0.5,
            'early_stopping_rounds': 100
        }
        
        model = CatBoostClassifier(**params)
        
        model.fit(
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            verbose=100
        )
        oof_pred[val_ind] = model.predict_proba(x_val)
        y_pred += model.predict_proba(df_test[use_features]) / folds.n_splits
        
        y_one_hot = label_binarize(y_val, np.arange(4)) 
        oof_one_hot = label_binarize(oof_pred[val_ind].argmax(axis=1), np.arange(4)) 
        score = roc_auc_score(y_one_hot, oof_one_hot) 
        print('auc: ', score)
        
        del x_train, x_val, y_train, y_val
        gc.collect()
    
    return y_pred, oof_pred

y_pred, oof_pred = run_nn(train_X, test_X, y)

y_one_hot = label_binarize(train['label'], np.arange(4)) 
oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
score = roc_auc_score(y_one_hot, oof_one_hot) 
print('auc: ', score)

#################################### 第五部分 ####################################
## submission

test = pd.read_pickle('test_04j20p_fe_v3.pickle')[['event_id']]
test["label"]= y_pred.argmax(axis=1)
test["label"] = test["label"].map({0:21, 1:1, 2:4, 3:5})

submission = pd.read_csv('./jet_complex_data/complex_test_R04_jet.csv')[['jet_id', 'event_id']]
submission = pd.merge(submission, test, on=['event_id'])
submission = submission.drop(['event_id'], axis=1)
submission = submission.rename(columns={'jet_id': 'id'})
print(submission.head())

submission.to_csv('submission_cat_baseline.csv', index=False)

np.save('y_pred_cat', y_pred)
np.save('oof_pred_cat', oof_pred)