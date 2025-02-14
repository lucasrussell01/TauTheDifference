# populate the configs for different eras, extracting automatically from HiggsDNA

import numpy as np
from glob import glob
import os
import json
import yaml
from utils import get_logger
from selection import Selector
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Extract parameters from HiggsDNA configs")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()

args = get_args()
logger = get_logger(debug=args.debug)


def update_config(cfg):
    print(cfg)


def main():
    eras = ['Run3_2022', 'Run3_2022EE', 'Run3_2023', 'Run3_2023BPix']
    path_to_HiggsDNA = '/vols/cms/lcr119/offline/HiggsCP/HiggsDNA/scripts/ditau/config'
    for era in eras:
        print(f'\nUpdating config for era: {era}')
        # Load configuration for the desired era
        cfg_in = yaml.safe_load(open(f"{path_to_HiggsDNA}/{era}/params.yaml"))
        cfg_out = yaml.safe_load(open(f"../config/{era}.yaml"))
        # Load processes:
        for process in cfg_out['Process']:
            if 'DATA' not in process:
                print(f"Process: {process}")
                for ds in cfg_out['Process'][process]:
                    print(ds)
                    cfg_out['Process'][process][ds]['x_sec'] = cfg_in[ds]['xs']
                    cfg_out['Process'][process][ds]['n_eff'] = cfg_in[ds]['eff']
                    cfg_out['Process'][process][ds]['filter_eff'] = cfg_in[ds]['filter_efficiency']
                # for ggH we need to sum everything up
                if process == 'ggH':
                    neff_sum = sum([cfg_out['Process'][process][ds]['n_eff'] for ds in cfg_out['Process'][process]]) # total eff across the ggH samples
                    for ds in cfg_out['Process'][process]:
                        cfg_out['Process'][process][ds]['n_eff'] = neff_sum # set the neff to be the sum


        save_path = f"../config/{era}.yaml"
        with open(save_path, 'w') as file:
            documents = yaml.dump(cfg_out, file)


if __name__ == "__main__":
    main()