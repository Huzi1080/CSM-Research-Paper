import os

print("=== Starting Adversarial Robustness Pipeline ===")

os.system("python scripts/download_nsl_kdd.py")
os.system("python scripts/train_baseline.py")
os.system("python scripts/attacks_numpy.py")
os.system("python scripts/analyze_results.py")

print("=== Pipeline complete! Check results/ folder for outputs. ===")
