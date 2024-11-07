import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Submit condor jobs multiple times.')
    parser.add_argument('--n_runs', type=int, help='Number of times to submit the job')
    parser.add_argument('--condor_sub', type=str, help='Path to the condor submit file')
    return parser.parse_args()


def submit_jobs(n_runs, condor_sub_file):
    for _ in range(n_runs):
        cmd = f'condor_submit {condor_sub_file}'
        print(f'Running: {cmd}')
        os.system(cmd)

if __name__ == "__main__":
    args = get_args()
    submit_jobs(args.n_runs, args.condor_sub)