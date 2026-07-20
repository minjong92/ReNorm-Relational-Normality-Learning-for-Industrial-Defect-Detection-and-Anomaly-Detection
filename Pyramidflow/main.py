import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from pathfilemgr import MPathFileManager


LOSSES = [1]
ALPHAS = [0.3]
BETAS = [1]
SEEDS = [0]
GRID_MODE = 'full'

parser = argparse.ArgumentParser()
parser.add_argument("--volume", default="moai")
parser.add_argument("--project", default="test_project")
parser.add_argument("--subproject", default="test_subproject")
parser.add_argument("--task", default="test_task")
parser.add_argument("--version", default="matrix", help="base output version folder")
parser.add_argument("--dataset", default="mvtec_anomaly_detection")
parser.add_argument("--category", default="all")
parser.add_argument("--dataset-root", default=r"E:\0_dataset")
parser.add_argument("--epoches", type=int, default=50)
parser.add_argument("--batchSize", "--batchsize", dest="batchSize", type=int, default=4)
parser.add_argument(
    "--grid_mode",
    default=GRID_MODE,
    choices=["paired", "full"],
    help="paired runs alpha=beta pairs only; full runs all alpha/beta combinations.",
)
parser.add_argument("--dry-run", action="store_true", help="print commands without running train.py")

opt = parser.parse_args()


MANIFEST_FIELDS = [
    "started_at",
    "finished_at",
    "status",
    "returncode",
    "duration_sec",
    "train_started_at",
    "train_finished_at",
    "train_status",
    "train_returncode",
    "train_duration_sec",
    "test_started_at",
    "test_finished_at",
    "test_status",
    "test_returncode",
    "test_duration_sec",
    "experiment",
    "loss",
    "alpha",
    "beta",
    "seed",
    "dataset",
    "category",
    "version",
    "output_dir",
    "train_command",
    "test_command",
]


def format_float(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def build_version(base_version, loss, alpha, beta, seed):
    name = f"loss{loss}_{format_float(alpha)}_{format_float(beta)}_{seed}"
    if base_version:
        return f"{base_version}/{name}"
    return name


def now_iso():
    return datetime.now().replace(microsecond=0).isoformat()


def resolve_output_dir(args, version):
    mpfm = MPathFileManager(args.volume, args.project, args.subproject, args.task, version)
    return mpfm.version_path


def manifest_path_for(args):
    mpfm = MPathFileManager(args.volume, args.project, args.subproject, args.task, args.version)
    return os.path.join(mpfm.version_path, "experiments_manifest.csv")


def write_manifest(path, rows):
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def iter_alpha_beta_pairs(grid_mode):
    if grid_mode == "paired":
        return ((alpha, alpha) for alpha in ALPHAS)
    return itertools.product(ALPHAS, BETAS)


def build_commands(args):
    train_py = Path(__file__).with_name("train.py")
    test_py = Path(__file__).with_name("test.py")
    commands = []

    for loss in LOSSES:
        for alpha, beta in iter_alpha_beta_pairs(args.grid_mode):
            for seed in SEEDS:
                version = build_version(
                    args.version,
                    loss=loss,
                    alpha=alpha,
                    beta=beta,
                    seed=seed,
                )
                command = [
                    sys.executable,
                    str(train_py),
                    "--volume",
                    args.volume,
                    "--project",
                    args.project,
                    "--subproject",
                    args.subproject,
                    "--task",
                    args.task,
                    "--version",
                    version,
                    "--dataset",
                    args.dataset,
                    "--category",
                    args.category,
                    "--dataset-root",
                    args.dataset_root,
                    "--seed",
                    str(seed),
                    "--loss",
                    str(loss),
                    "--alpha",
                    str(alpha),
                    "--beta",
                    str(beta),
                ]

                if args.epoches is not None:
                    command.extend(["--epoches", str(args.epoches)])
                if args.batchSize is not None:
                    command.extend(["--batchSize", str(args.batchSize)])

                test_command = [
                    sys.executable,
                    str(test_py),
                    "--volume",
                    args.volume,
                    "--project",
                    args.project,
                    "--subproject",
                    args.subproject,
                    "--task",
                    args.task,
                    "--version",
                    version,
                    "--dataset",
                    args.dataset,
                    "--category",
                    args.category,
                    "--dataset-root",
                    args.dataset_root,
                ]

                commands.append({
                    "started_at": "",
                    "finished_at": "",
                    "status": "pending",
                    "returncode": "",
                    "duration_sec": "",
                    "train_started_at": "",
                    "train_finished_at": "",
                    "train_status": "pending",
                    "train_returncode": "",
                    "train_duration_sec": "",
                    "test_started_at": "",
                    "test_finished_at": "",
                    "test_status": "pending",
                    "test_returncode": "",
                    "test_duration_sec": "",
                    "experiment": version.replace("\\", "/").split("/")[-1],
                    "loss": loss,
                    "alpha": alpha,
                    "beta": beta,
                    "seed": seed,
                    "dataset": args.dataset,
                    "category": args.category,
                    "version": version,
                    "output_dir": resolve_output_dir(args, version),
                    "train_command": " ".join(command),
                    "test_command": " ".join(test_command),
                    "_train_command_list": command,
                    "_test_command_list": test_command,
                })

    return commands


def run_step(row, rows, manifest_path, step_name, command):
    row[f"{step_name}_status"] = "running"
    row[f"{step_name}_started_at"] = now_iso()
    row["status"] = f"{step_name}_running"
    write_manifest(manifest_path, rows)

    start = time.perf_counter()
    result = subprocess.run(command)
    duration = time.perf_counter() - start

    row[f"{step_name}_finished_at"] = now_iso()
    row[f"{step_name}_returncode"] = result.returncode
    row[f"{step_name}_duration_sec"] = f"{duration:.3f}"
    row[f"{step_name}_status"] = "ok" if result.returncode == 0 else "failed"
    write_manifest(manifest_path, rows)
    return result


def main():
    args = opt
    rows = build_commands(args)
    manifest_path = manifest_path_for(args)
    write_manifest(manifest_path, rows)

    print(f"Total runs: {len(rows)}")
    print(f"Manifest: {manifest_path}")

    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] version={row['version']}")
        print(f"train: {row['train_command']}")
        print(f"test : {row['test_command']}")
        if args.dry_run:
            row["status"] = "dry_run"
            row["train_status"] = "dry_run"
            row["test_status"] = "dry_run"
            write_manifest(manifest_path, rows)
            continue

        row["status"] = "running"
        row["started_at"] = now_iso()
        write_manifest(manifest_path, rows)

        start = time.perf_counter()
        train_result = run_step(row, rows, manifest_path, "train", row["_train_command_list"])
        if train_result.returncode != 0:
            duration = time.perf_counter() - start
            row["finished_at"] = now_iso()
            row["returncode"] = train_result.returncode
            row["duration_sec"] = f"{duration:.3f}"
            row["status"] = "failed"
            write_manifest(manifest_path, rows)
            raise subprocess.CalledProcessError(train_result.returncode, row["_train_command_list"])

        test_result = run_step(row, rows, manifest_path, "test", row["_test_command_list"])
        duration = time.perf_counter() - start

        row["finished_at"] = now_iso()
        row["returncode"] = test_result.returncode
        row["duration_sec"] = f"{duration:.3f}"
        row["status"] = "ok" if test_result.returncode == 0 else "failed"
        write_manifest(manifest_path, rows)

        if test_result.returncode != 0:
            raise subprocess.CalledProcessError(test_result.returncode, row["_test_command_list"])


if __name__ == "__main__":
    main()
