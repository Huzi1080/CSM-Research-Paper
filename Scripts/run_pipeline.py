# run_pipeline.py
"""
Runs the full research pipeline:
1. Downloads & preprocesses dataset
2. Trains baseline ML model
3. Performs adversarial attacks (FGSM & PGD)
4. Generates evaluation plots and summary report

Usage:
    python run_pipeline.py
"""

import os
import subprocess
import time

SCRIPTS = [
    "scripts/download_nsl_kdd.py",
    "scripts/train_baseline.py",
    "scripts/attacks_numpy.py",
    "scripts/analyze_results.py"
]

def run_step(script_path):
    print("\n" + "="*70)
    print(f"▶ Running {script_path} ...")
    print("="*70)
    start = time.time()
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {script_path}: {e}")
        raise
    end = time.time()
    print(f"✅ Completed {script_path} in {end - start:.1f} seconds")

def main():
    print("\n🚀 Starting full research pipeline...\n")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("attacks", exist_ok=True)

    for script in SCRIPTS:
        if not os.path.exists(script):
            print(f"⚠️ Skipping missing script: {script}")
            continue
        run_step(script)

    print("\n🎯 Pipeline finished successfully!")
    print("📊 Results: see the 'results/' folder for plots and summary_report.md")
    print("🧠 Model: stored in 'models/baseline_model.pkl'")
    print("💥 Adversarial samples: stored in 'attacks/'")

if __name__ == "__main__":
    main()
