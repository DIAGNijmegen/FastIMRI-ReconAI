import sys
import json
from pathlib import Path


def main():

    folder_ref = '../../../segmentation/test/recon_crnn_ne_8_64_5_seg/summary.json'



    with open(Path(folder_ref).resolve()) as json_data:
        summary_json = json.load(json_data)

        for case in summary_json["metric_per_case"]:
            print(case["metrics"]["(1,)"]["Dice"])


    "metric_per_case"



if __name__ == '__main__':
    main()
