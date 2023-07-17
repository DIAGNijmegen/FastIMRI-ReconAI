from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, labels_to_list_of_regions

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import sys


def main():
    folder_ref = '../../../segmentation/test/gnd_seg'

    folder_pred = sys.argv[1]
    output_file = f'{folder_pred}/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions,
                              ignore_label, num_processes, only_on_dim=2)


if __name__ == '__main__':
    main()