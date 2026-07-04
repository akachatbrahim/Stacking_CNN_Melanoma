import os
import subprocess
import sys
from pathlib import Path

import joblib

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_INDEX = 43

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def list_available_models(models_dir=None):
    if models_dir is None:
        models_dir = PROJECT_DIR / "Models"

    return sorted([p for p in Path(models_dir).rglob("*.pkl") if p.is_file()])


def get_default_model_path(models_dir=None):
    model_files = list_available_models(models_dir)
    if not model_files:
        raise FileNotFoundError("No .pkl model files were found under the Models folder.")

    if len(model_files) > DEFAULT_MODEL_INDEX:
        return model_files[DEFAULT_MODEL_INDEX]

    return model_files[-1]


def show_menu():
    print("=" * 60)
    print("Melanoma Stacking Project Menu")
    print("=" * 60)
    print("1. Run the stacking model training and evaluation.")
    print("2. Generate the experimental results report.")
    print("3. Generate the performance plots and visualizations.")
    print("4. Image Diagnostic")
    print("0. Exit")
    print("=" * 60)


def run_script(script_name: str, args=None):
    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        print(f"\n❌ Script not found: {script_path}")
        return

    print(f"\n▶ Running {script_name}...")
    print("-" * 60)
    command = [sys.executable, str(script_path)]
    if args:
        command.extend(args)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    result = subprocess.run(command, cwd=str(PROJECT_DIR), env=env, stderr=subprocess.DEVNULL)
    print("-" * 60)
    if result.returncode == 0:
        print(f"✅ Finished {script_name}")
    else:
        print(f"⚠️ {script_name} exited with code {result.returncode}")


def handle_plotting_menu():
    print("\n📊 Visualization options:")
    print("1. Default Model Evaluation (Model melanoma_stacking_safe_lr3e05_dw1e05)")
    print("2. Selected Model Evaluation")
    print("0. Return to the main menu")
    try:
        plot_choice = input("Choose an option (0-2): ").strip() or "1"
    except EOFError:
        print("\nNo input received. Exiting.")
        raise SystemExit(0)

    if plot_choice == "1":
        run_script("plots_melanoma_stacking.py", ["--mode", "default"])
    elif plot_choice == "2":
        run_script("plots_melanoma_stacking.py", ["--mode", "select"])
    elif plot_choice == "0":
        print("\nReturning to the main menu.")
    else:
        print("\nInvalid plotting choice. Please enter 0, 1, or 2.")


def main():
    while True:
        clear_screen()

        show_menu()
        try:
            choice = input("Choose an option (0-4): ").strip()
        except EOFError:
            print("\nNo input received. Exiting.")
            break

        if choice == "1":
            run_script("melanoma_stacking.py")
        elif choice == "2":
            run_script("pkl_to_word_report.py")
        elif choice == "3":
            handle_plotting_menu()
        elif choice == "4":
            run_script("image_prediction.py")
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 0, 1, 2, 3, or 4.")

        try:
            input("\nPress Enter to continue...")
        except EOFError:
            print("\nNo input received. Exiting.")
            break


if __name__ == "__main__":
    main()
