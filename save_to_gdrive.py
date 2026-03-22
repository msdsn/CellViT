"""Periodically save training outputs to Google Drive."""

import time
import os
import shutil
import subprocess
import argparse


def find_run_dir(log_dir):
    """Find the latest run directory under log_dir."""
    dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("2")]
    if not dirs:
        return None
    return os.path.join(log_dir, sorted(dirs)[-1])


def get_current_epoch(log_file):
    """Get current epoch from training log."""
    try:
        r = subprocess.run(["grep", "Epoch:", log_file], capture_output=True, text=True)
        lines = r.stdout.strip().split("\n")
        return lines[-1].split("Epoch: ")[1].split("/")[0]
    except:
        return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="CellViT log directory (e.g. ./logs_paper/PanNuke)")
    parser.add_argument("--gdrive_dir", type=str, required=True, help="Google Drive save directory")
    parser.add_argument("--interval", type=int, default=1800, help="Save interval in seconds (default: 1800)")
    args = parser.parse_args()

    os.makedirs(args.gdrive_dir, exist_ok=True)
    print("Checkpoint saver started")
    print("  Log dir: %s" % args.log_dir)
    print("  GDrive: %s" % args.gdrive_dir)
    print("  Interval: %d seconds" % args.interval)

    while True:
        time.sleep(args.interval)

        run_dir = find_run_dir(args.log_dir)
        if not run_dir:
            print("No run directory found, waiting...")
            continue

        log_file = os.path.join(run_dir, "logs.log")
        ep = get_current_epoch(log_file)

        # Copy config and logs
        for f in ["config.yaml", "logs.log"]:
            src = os.path.join(run_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.gdrive_dir, f))

        # Copy best checkpoint
        best = os.path.join(run_dir, "checkpoints", "model_best.pth")
        if os.path.exists(best):
            shutil.copy2(best, os.path.join(args.gdrive_dir, "model_best.pth"))

        print("[%s] Epoch %s saved" % (time.strftime("%H:%M"), ep), flush=True)

        # Check if training is still running
        r = subprocess.run(["pgrep", "-f", "run_cellvit"], capture_output=True)
        if r.returncode != 0:
            latest = os.path.join(run_dir, "checkpoints", "latest_checkpoint.pth")
            if os.path.exists(latest):
                shutil.copy2(latest, os.path.join(args.gdrive_dir, "latest_checkpoint.pth"))
            print("Training finished! All saved.")
            break


if __name__ == "__main__":
    main()
