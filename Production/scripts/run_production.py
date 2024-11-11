from glob import glob
import os
import json
import yaml
import argparse
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description="Run Production for Classifier Samples")
    parser.add_argument('--channel', type=str, help="Channel to process", required=True)
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--extrapolate', action='store_true', help="Extrapolate QCD")
    parser.add_argument('--cut', type=str, help="Cut for tt channel", required=False)
    return parser.parse_args()

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Command '{command}' ran successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with exit code: {e.returncode}")
        raise

def run_production(channel, debug=False, cut=None):
    # Run standard production (PreSelect, Process, ShuffleMerge)
    debug_flag = "--debug" if debug else ""
    run_command(f"python ../python/PreSelect.py --channel {channel} {debug_flag}")
    run_command(f"python ../python/Process.py --channel {channel} {debug_flag} --cut {cut}")
    run_command(f"python ../python/ShuffleMerge.py --channel {channel} {debug_flag}")

def extrapolate_QCD(channel, debug=False):
    # Get QCD estimates (PreSelect, Process, extrapolateQCD)
    debug_flag = "--debug" if debug else ""
    run_command(f"python ../python/PreSelect.py --channel {channel} --extrapolate {debug_flag}")
    run_command(f"python ../python/Process.py --channel {channel} --extrapolate {debug_flag}")
    run_command(f"python ../python/extrapolateQCD.py --channel {channel} {debug_flag}")

def main():
    args = get_args()
    try:
        if args.extrapolate:
            extrapolate_QCD(args.channel, args.debug)
        else:
            run_production(args.channel, args.debug, args.cut)
    except subprocess.CalledProcessError:
        raise RuntimeError("Production failed!")

if __name__ == "__main__":
    main()
