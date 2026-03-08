# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import pandas as pd
import os

class LoadProfile:
    def __init__(self, steps, dss_folder_path, dss_file, worker_idx=None, run_token=None):
        self.steps = steps
        
        self.dss_folder_path = dss_folder_path
        self.loadshape_path = os.path.join(dss_folder_path, 'loadshape')
        self.run_token = run_token
        active_dir_parts = [dss_folder_path, 'loadshape_active']
        if self.run_token is not None:
            active_dir_parts.append(self.run_token)
        self.active_profile_dir = os.path.join(*active_dir_parts)
        os.makedirs(self.active_profile_dir, exist_ok=True)
        if worker_idx is None:
            self.loadshape_dss_name = 'loadshape.dss'
            self.active_profile_csv = 'active_load_profile.csv'
        else:
            self.loadshape_dss_name = 'loadshape_' + str(worker_idx) + '.dss'
            self.active_profile_csv = 'active_load_profile_' + str(worker_idx) + '.csv'
        redirect_parts = ['loadshape_active']
        if self.run_token is not None:
            redirect_parts.append(self.run_token)
        redirect_parts.append(self.loadshape_dss_name)
        self.loadshape_dss = os.path.join(*redirect_parts)
        
        self.LOAD_NAMES = self.find_load_names(dss_file)
        self._profiles = None
        self._num_profiles = None
        self._scale = None
    
        self.FILES = []
        for f in os.listdir(self.loadshape_path):
            low = f.lower()
            if ('loadshape' in low) and low.endswith('.csv'):
                self.FILES.append(os.path.join(self.loadshape_path, f))
        

    def create_file_with_daily(self, fname):
        '''
            Create a new file named fname[:-4]_daily.dss
            In the new file,
            if any load is created, then the load is associated with its daily loadshape.
        '''
        fin = open(os.path.join(self.dss_folder_path, fname), 'r')
        fout = open(os.path.join(self.dss_folder_path, fname[:-4] + '_daily.dss'), 'w')
        for line in fin:
            if not line.lower().startswith('new load.') or ('daily' in line):
                fout.write(line)
            else:
                line = line.strip()
                if '!' in line: line = line[:line.find('!')].strip() # remove inline comment
                if '//' in line: line = line[:line.find('//')].strip() # remove inline comment
                spt = list(filter(None, line.split(' '))) # filter out the empty string
                load = spt[1].split('.',1)[1]
                fout.write(line + ' daily=loadshape_' + load + '\n')
        fin.close()
        fout.close()
   
    def add_redirect_and_mode_at_main_daily_dss(self, main_daily_dss):
        '''
            Add redirect loadshape (& load file if any) 
            and set daily mode at the main daily dss file

        Args:
            main_daily_dss: the file name of the main daily dss file
        
        Returns:
            the load dss file (if any) associated with the main dss file
        '''
        # load the file
        fin = open(os.path.join(self.dss_folder_path, main_daily_dss), 'r')
        lines = [line for line in fin]
        fin.close()

        # overwrite the file
        found_load, redirect_load = False, False
        load_file = None
        fout = open(os.path.join(self.dss_folder_path, main_daily_dss), 'w')
        for line in lines:
            low = line.strip().lower()
            if '!' in low: low = low[:low.find('!')].strip() # remove inline comment
            if '//' in low: low = low[:low.find('//')].strip() # remove inline comment
            if (not found_load) and 'load' in low and not low.startswith('~'):
                fout.write('! add loadshape\n')
                fout.write('redirect ' + self.loadshape_dss + '\n\n')
                found_load = True

            low = low[:-4] if len(low)>=4 else ''
            if (not redirect_load) and low.startswith('redirect'):
                if low.endswith('loads') or low.endswith('load'):
                    load_file = list(filter(None, line.strip().split(' ')))[1] # remove the empty string
                    fout.write('redirect ' + load_file[:-4] + '_daily.dss\n')
                    redirect_load = True
                elif low.endswith('loads_daily') or low.endswith('load_daily'):
                    load_file = list(filter(None, line.strip().split(' ')))[1] # remove the empty string
                    fout.write(line)
                    redirect_load = True
                else: fout.write(line)
            else: fout.write(line)
        
        assert found_load, 'cannot find load at ' + main_daily_dss

        fout.write('Set mode=Daily number=1 hour=0 stepsize=3600 sec=0\n')
        fout.close()
        
        return load_file

    def ensure_loadshape_redirect(self, main_dss):
        file_path = os.path.join(self.dss_folder_path, main_dss)
        with open(file_path, 'r') as fin:
            lines = fin.readlines()

        updated = False
        with open(file_path, 'w') as fout:
            for line in lines:
                stripped = line.strip()
                low = stripped.lower()
                if low.startswith('redirect '):
                    target = stripped.split(None, 1)[1]
                    target_name = os.path.basename(target).lower()
                    if target_name.startswith('loadshape') and target_name.endswith('.dss'):
                        line = 'redirect ' + self.loadshape_dss + '\n'
                        updated = True
                fout.write(line)

        return updated

    def find_load_file_from(self, main_dss):
        fin = open(os.path.join(self.dss_folder_path, main_dss), 'r')
        load_file = None
        for line in fin:
            low = line.strip().lower()
            if '!' in low: low = low[:low.find('!')].strip() # remove inline comment
            if '//' in low: low = low[:low.find('//')].strip() # remove inline comment
            low = low[:-4] if len(low)>=4 else ''
            if low.startswith('redirect') and \
               (low.endswith('loads') or low.endswith('load') \
                or low.endswith('loads_daily') or low.endswith('load_daily') ):
                
                load_file = list(filter(None, line.strip().split(' ')))[1]
                break
        return load_file

    def find_load_names(self, main_dss):
        '''
            Find the loads with daily loadshapes at main dss or the load dss files.
            If there is none, 
            then generate new files (annotated _daily) with daily loadshapes.
        '''
        def find_load_name(fname, names):
            file_path = os.path.join(self.dss_folder_path, fname)
            assert os.path.exists(file_path), file_path + ' not found'
            
            needs_load_daily, daily_mode = False, False
            with open(file_path, 'r') as fin:
                for line in fin:
                    low = line.strip().lower()
                    if low.startswith('new load.'):
                        if 'daily' in low:
                            spt = line.split(' ')
                            spt = list(filter(None, spt)) # filter out the empty string
                            names.append(spt[1].split('.',1)[1])
                        else: needs_load_daily = True
                    if low.startswith('set mode=daily'):
                        daily_mode = True
            return needs_load_daily, daily_mode
        names = []

        # add from the main dss file
        needs_load_daily, daily_mode = find_load_name(main_dss, names)
        if needs_load_daily or (not daily_mode):
            ## Create a new _daily file. Add daily loadshape if needed
            self.create_file_with_daily(main_dss)

            ## add redirect and set daily mode at the new _daily file
            load_file = self.add_redirect_and_mode_at_main_daily_dss(\
                                              main_dss[:-4]+'_daily.dss')
        else:
            self.ensure_loadshape_redirect(main_dss)
            load_file = self.find_load_file_from(main_dss)

        # add from the other load files
        if load_file is not None:
            if 'daily' in load_file:
                needs_load_daily, _ = find_load_name(load_file, names)
                assert (not needs_load_daily), 'invalid content in ' + load_file
            else:
                needs_load_daily, _ = find_load_name(load_file, names)
                if needs_load_daily: self.create_file_with_daily(load_file)

        # check empty or duplicate load
        assert len(names)>0, 'daliy load not found. Consider modifying from the auto-generated file annotated with _daily'
        assert len(names) == len(set(names)), 'duplicate load names'
        
        return names     

    def _load_profiles(self, scale=1.0):
        if self._profiles is not None and self._scale == scale:
            return self._profiles, self._num_profiles

        try:
            dfs = []
            for f in self.FILES:
                dfs.append(pd.read_csv(f, header=None))
            assert len(dfs)>0, r'put load shapes files under ./loadshape'
            values = pd.concat(dfs).rename(columns={0: 'mul'}).reset_index(drop=True)['mul']
            if scale != 1.0:
                values = values * scale
        except Exception as exc:
            raise RuntimeError(r'put load shapes files under ./loadshape') from exc

        episodes = len(values) // (self.steps * len(self.LOAD_NAMES))
        assert episodes > 0, 'insufficient loadshape data'

        total_points = self.steps * episodes * len(self.LOAD_NAMES)
        values = values.iloc[:total_points].to_numpy()

        profiles = {}
        per_load_points = self.steps * episodes
        offset = 0
        for load_name in self.LOAD_NAMES:
            chunk = values[offset:offset + per_load_points]
            profiles[load_name] = chunk.reshape(episodes, self.steps)
            offset += per_load_points

        self._profiles = profiles
        self._num_profiles = episodes
        self._scale = scale
        return profiles, episodes

    def gen_loadprofile(self, scale=1.0):
        _, episodes = self._load_profiles(scale=scale)
        return episodes # number of distinct epochs
    
    def choose_loadprofile(self, idx):
        profiles, episodes = self._load_profiles(scale=self._scale if self._scale is not None else 1.0)
        assert 0 <= idx < episodes, 'idx does not exist'

        active_profile = pd.DataFrame({
            load_name: profiles[load_name][idx] for load_name in self.LOAD_NAMES
        })
        active_profile_path = os.path.join(self.active_profile_dir, self.active_profile_csv)
        active_profile.to_csv(active_profile_path, header=False, index=False)

        with open(os.path.join(self.active_profile_dir, self.loadshape_dss_name), 'w') as fp:
            for col_idx, load_name in enumerate(self.LOAD_NAMES, start=1):
                fp.write(f'New Loadshape.loadshape_{load_name} npts={self.steps} sinterval={60*60*24//self.steps} ' +
                    f'mult=(file=./{self.active_profile_csv}, col={col_idx}, header=no)\n')

        return active_profile_path

    def get_loadprofile(self, idx):
        profiles, episodes = self._load_profiles(scale=self._scale if self._scale is not None else 1.0)
        assert 0 <= idx < episodes, 'idx does not exist'
        return pd.DataFrame({
            load_name: profiles[load_name][idx] for load_name in self.LOAD_NAMES
        })
