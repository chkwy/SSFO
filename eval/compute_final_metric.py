import os
import json
import pandas as pd


def extract_metrics_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            "sub_EM": data.get("sub_EM", None),
            "sub_Subspan_EM": data.get("sub_Subspan_EM", None),
            "org_EM": data.get("org_EM", None),
            "org_Subspan_EM": data.get("org_Subspan_EM", None),
            "rouge-1": data.get("rouge-1", {}).get("f", None),
            "rouge-2": data.get("rouge-2", {}).get("f", None),
            "rouge-l": data.get("rouge-l", {}).get("f", None),
        }


def main():
    output_folder = '/data/DFO/eval/outputs'
    metrics = []

    for root, _, files in os.walk(output_folder):
        for file in files:
            if file.endswith('metrics_final.json'):
                file_path = os.path.join(root, file)
                metrics_data = extract_metrics_from_json(file_path)
                metrics_data["model"] = file_path.split('/')[-1].split('_')[1]
                if metrics_data["model"] == "trustalign" or metrics_data["model"] == "test":
                    metrics_data["model"] = "_".join(
                        file_path.split('/')[-1].split('_')[1:4])
                metrics_data["task"] = file_path.split('/')[-1].split('_')[0]
                metrics.append(metrics_data)

    df = pd.DataFrame(metrics)
    df.to_excel('/data/DFO/eval/summary_metrics.xlsx', index=False)


if __name__ == "__main__":
    main()
