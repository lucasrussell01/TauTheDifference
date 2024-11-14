import optuna
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for XGBoost")
    parser.add_argument('--study_name', type=str, help="Name of study to check")
    return parser.parse_args()

def main(study_name):
    none_counts = 0
    valid_counts = 0

    # Load study
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///hyperlogs/{study_name}.db?timeout=10000")

    print(f"Study name: {study_name}")

    # Check all trials for stats
    for trial in study.trials:
        # print(f"Trial #{trial.number}, Value: {trial.value}, Params: {trial.params}")
        if trial.value is None:
            none_counts += 1
        else:
            valid_counts += 1

    print(f"Number of trials with None values: {none_counts}")
    print(f"Number of trials with valid values: {valid_counts}")
    print('-'*140)

    trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    '{key}': {value}")

if __name__ == "__main__":
    args = get_args()
    main(args.study_name)
