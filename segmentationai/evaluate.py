from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, labels_to_list_of_regions

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def main():
    folder_ref = '../../../segmentation/nnUNet_raw/Dataset502_fastmri_intervention/labelsTr'
    folder_pred = '../../../segmentation/test/recon_ne_16_64_5_seg'
    output_file = '../../../segmentation/test/recon_ne_16_64_5_seg/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions,
                              ignore_label,
                              num_processes)


if __name__ == '__main__':
    main()