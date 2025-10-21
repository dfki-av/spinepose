import json
import os



def get_results(results_dir) -> dict:
    """Find all JSON result files in the specified directory."""
    # Find all JSON files in this, including subdirectory, then read and return the last one
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                result_files.append(os.path.join(root, file_name))
    if not result_files:
        return None

    # Read the last JSON file
    with open(result_files[-1], 'r') as f:
        return json.load(f)


def summarize_results(results) -> dict:
    metrics = [
        "spinetrack/body_AP",
        "spinetrack/body_AR",
        "spinetrack/feet_AP",
        "spinetrack/feet_AR",
        "spinetrack/spine_AP",
        "spinetrack/spine_AR",
    ]

    summary = {}
    for model, model_results in results.items():
        summary[model] = {}
        for metric in metrics:
            if metric in model_results:
                summary[model][metric] = model_results[metric]
            else:
                summary[model][metric] = None
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize SpineTrack results from multiple models"
    )
    parser.add_argument(
        "--results_dirs",
        type=str,
        default="work_dirs",
        help="Directory containing subdirectories of model results",
    )
    args = parser.parse_args()
    results_dirs = args.results_dirs

    models = [
        "spinepose-s_32xb256-10e_spinetrack-256x192",
        "spinepose-m_32xb256-10e_spinetrack-256x192",
        "spinepose-l_32xb256-10e_spinetrack-256x192",
        "spinepose-x_32xb128-10e_spinetrack-384x288",
    ]
    all_results = {}
    for model in models:
        model_dir = os.path.join(results_dirs, model)
        result = get_results(model_dir)
        if result is not None:
            all_results[model] = result
    summary = summarize_results(all_results)

    header = ["Model", "Body AP", "Body AR", "Feet AP", "Feet AR", "Spine AP", "Spine AR"]
    print("{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(*header))
    print("-" * 80)
    for model, metrics in summary.items():
        print(
            "{:<20}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(
                model.split("_")[0],
                f"{metrics['spinetrack/body_AP'] * 100:.2f}"
                if metrics['spinetrack/body_AP'] is not None
                else "N/A",
                f"{metrics['spinetrack/body_AR'] * 100:.2f}"
                if metrics['spinetrack/body_AR'] is not None
                else "N/A",
                f"{metrics['spinetrack/feet_AP'] * 100:.2f}"
                if metrics['spinetrack/feet_AP'] is not None
                else "N/A",
                f"{metrics['spinetrack/feet_AR'] * 100:.2f}"
                if metrics['spinetrack/feet_AR'] is not None
                else "N/A",
                f"{metrics['spinetrack/spine_AP'] * 100:.2f}"
                if metrics['spinetrack/spine_AP'] is not None
                else "N/A",
                f"{metrics['spinetrack/spine_AR'] * 100:.2f}"
                if metrics['spinetrack/spine_AR'] is not None
                else "N/A",
            )
        )

