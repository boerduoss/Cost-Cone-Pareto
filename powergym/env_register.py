# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import os
import inspect
import re
import shutil
from powergym.env import Env

# map from system_name to fixed information of the system

_SYS_INFO = {
    '13Bus': {
        'source_bus': 'sourcebus',
        'node_size': 500,
        'shift': 10,
        'show_node_labels': True
    },

    '34Bus': {
        'source_bus': 'sourcebus',
        'node_size': 500,
        'shift': 80,
        'show_node_labels': True
    },

    '123Bus': {
        'source_bus': '150',
        'node_size': 400,
        'shift': 80,
        'show_node_labels': True
    },

    '8500-Node': {
        'source_bus': 'e192860',
        'node_size': 10,
        'shift': 0,
        'show_node_labels': False
    }
}


# map from env_name to the necessary information
_ENV_INFO = {
    '13Bus': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 6.0/33
    },
   
    '13Bus_cbat': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 6.0/33,
    },
   
    '13Bus_soc': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 20.0/33,
        'dis_w': 1.0/33
    },

    '13Bus_cbat_soc': {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 20.0/33,
        'dis_w': 1.0/33,
    },

    '34Bus': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
    },

    '34Bus_cbat': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 10.0/33,
    },

    '34Bus_soc': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 4.0/33,
    },

    '34Bus_cbat_soc': {
        'system_name': '34Bus',
        'dss_file': 'ieee34Mod1_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 4.0/33,
    },

    '123Bus': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 7.0/33,
    },

    '123Bus_cbat': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 7.0/33,
    },

    '123Bus_soc': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 5.0/33,
    },

    '123Bus_cbat_soc': {
        'system_name': '123Bus',
        'dss_file': 'IEEE123Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 500.0/33,
        'dis_w': 5.0/33,
    },

    '8500Node': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 200.0/33,
    },
    
    '8500Node_cbat': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 200.0/33,
    },


    '8500Node_soc': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 10000/33,
        'dis_w': 100/33,
    },

    '8500Node_cbat_soc': {
        'system_name': '8500-Node',
        'dss_file': 'Master_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': float('inf'),
        'power_w': 1.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 10000/33,
        'dis_w': 100/33,
    }
}

# add system information to environment
for env in _ENV_INFO.keys():
    sys = _ENV_INFO[env]['system_name']
    _ENV_INFO[env].update(_SYS_INFO[sys])



####################### functions ########################
def get_info_and_folder(env_name):
    # check env scale and env name
    is_scaled = re.match('.*(_s)([0-9]*[.])?[0-9]+?', env_name)
    if is_scaled:
        matched_str = is_scaled.group(0)
        idx = matched_str.rfind('_s')
        env_name = matched_str[:idx]
        scale = float(matched_str[idx+2:])
    assert env_name in _ENV_INFO, env_name + ' not implemented'

    # get base_info
    base_info = _ENV_INFO[env_name].copy()
    if is_scaled:
        base_info['scale'] = scale
        base_info['soc_w'] = base_info['soc_w'] * (scale**2)

    # get folder path
    folder_path = os.path.join(os.path.dirname(os.path.abspath(inspect.getsourcefile(Env))), '..', 'systems')
    folder_path = os.path.abspath(folder_path)
    return base_info, folder_path

def _sanitize_run_token(run_token):
    if run_token is None:
        return None
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(run_token))

def _runtime_dss_name(base_name, worker_idx=None, run_token=None):
    if worker_idx is None and run_token is None:
        return base_name
    stem, ext = os.path.splitext(base_name)
    suffix = []
    if run_token is not None:
        suffix.append(run_token)
    if worker_idx is not None:
        suffix.append(str(worker_idx))
    return stem + '_' + '_'.join(suffix) + ext

def _loadshape_redirect_path(worker_idx=None, run_token=None):
    parts = ['loadshape_active']
    if run_token is not None:
        parts.append(run_token)
    if worker_idx is None:
        parts.append('loadshape.dss')
    else:
        parts.append('loadshape_' + str(worker_idx) + '.dss')
    return os.path.join(*parts)

def make_env(env_name, dss_act=False, worker_idx=None, run_token=None):
    base_info, folder_path = get_info_and_folder(env_name)
    run_token = _sanitize_run_token(run_token)

    if worker_idx is None and run_token is None:
        return Env(folder_path, base_info, dss_act)
    else:
        base_file = os.path.join(folder_path, base_info['system_name'], base_info['dss_file'])
        assert os.path.exists(base_file), base_file + ' does not exist'
        fin = open(base_file, 'r')

        runtime_dss_name = _runtime_dss_name(base_info['dss_file'], worker_idx=worker_idx, run_token=run_token)
        runtime_dss_path = os.path.join(folder_path, base_info['system_name'], runtime_dss_name)
        loadshape_redirect = _loadshape_redirect_path(worker_idx=worker_idx, run_token=run_token)

        with open(runtime_dss_path, 'w') as fout:
            for line in fin:
                stripped = line.strip()
                if stripped.startswith('redirect '):
                    target = stripped.split(None, 1)[1]
                    target_name = os.path.basename(target).lower()
                    if target_name.startswith('loadshape') and target_name.endswith('.dss'):
                        fout.write('redirect ' + loadshape_redirect + '\n')
                        continue
                if stripped == 'redirect loadshape.dss' or stripped == 'redirect loadshape_active/loadshape.dss':
                    fout.write('redirect ' + loadshape_redirect + '\n')
                else:
                    fout.write(line)
        info = base_info.copy()
        info['dss_file'] = runtime_dss_name
        if worker_idx is not None:
            info['worker_idx'] = worker_idx
        if run_token is not None:
            info['run_token'] = run_token
        return Env(folder_path, info, dss_act)
        
def remove_parallel_dss(env_name, num_workers, run_token=None):
    base_info, folder_path = get_info_and_folder(env_name)
    run_token = _sanitize_run_token(run_token)
    system_dir = os.path.join(folder_path, base_info['system_name'])

    if run_token is None:
        bases = [
            os.path.join(system_dir, base_info['dss_file']),
            os.path.join(system_dir, 'loadshape_active', 'loadshape.dss'),
        ]
        for base in bases:
            for i in range(num_workers):
                fname = base[:-4] + '_' + str(i) + '.dss'
                if os.path.exists(fname):
                    os.remove(fname)
        return

    for i in range(num_workers):
        main_name = _runtime_dss_name(base_info['dss_file'], worker_idx=i, run_token=run_token)
        main_path = os.path.join(system_dir, main_name)
        if os.path.exists(main_path):
            os.remove(main_path)

    active_dir = os.path.join(system_dir, 'loadshape_active', run_token)
    if os.path.isdir(active_dir):
        for entry in os.listdir(active_dir):
            os.remove(os.path.join(active_dir, entry))
        os.rmdir(active_dir)


def remove_runtime_artifacts(env_name, run_token):
    base_info, folder_path = get_info_and_folder(env_name)
    run_token = _sanitize_run_token(run_token)
    if run_token is None:
        return

    system_dir = os.path.join(folder_path, base_info['system_name'])
    stem, ext = os.path.splitext(base_info['dss_file'])
    prefix = stem + '_' + run_token
    for entry in os.listdir(system_dir):
        if entry.startswith(prefix) and entry.endswith(ext):
            os.remove(os.path.join(system_dir, entry))

    active_dir = os.path.join(system_dir, 'loadshape_active', run_token)
    if os.path.isdir(active_dir):
        shutil.rmtree(active_dir)
