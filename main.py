from evaluation import save_comparison_table, save_metrics_json
from preprocessing import load_and_preprocess_data
from training import run_full_training_pipeline


def main():
    print("\n--- LOADING & PREPROCESSING DATA ---")
    data = load_and_preprocess_data()
    print("Class distribution:", data["artifacts"]["class_distribution"])

    print("\n--- TRAINING & MODEL COMPARISON ---")
    _, best_metrics, comparison_rows, _ = run_full_training_pipeline(data)

    save_comparison_table(comparison_rows)
    save_metrics_json(best_metrics)

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print(f"Best test accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best test F1 (macro): {best_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
