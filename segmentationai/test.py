import os

if __name__ == '__main__':
    import torch
    from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import matplotlib.pyplot as plt
    import numpy as np

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder('../../../segmentation/nnUNet_results/Dataset500_fastmri_intervention/nnUNetTrainer__nnUNetPlans__3d_fullres', use_folds=(0,))
    # predictor.initialize_from_trained_model_folder('//chansey.umcn.nl\pelvis\projects\steffan\segmentation/nnUNet_results\Dataset500_fastmri_intervention/nnUNetTrainer__nnUNetPlans__3d_fullres', use_folds=None)

    img, props = SimpleITKIO().read_images(['6_4_0000.nii.gz'])
    img2, props2 = SimpleITKIO().read_images(['5_3_0000.nii.gz'])

    print(img.shape)
    exit(1)

    ret = predictor.predict_from_list_of_npy_arrays([img, img2],
                                                    None,
                                                    [props, props2],
                                                    None, 1, save_probabilities=True,
                                                    num_processes_segmentation_export=1)

    plt.imshow(np.abs(img[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.savefig('original1.png')
    plt.clf()

    plt.imshow(np.abs(img2[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.savefig('original2.png')
    plt.clf()

    result = np.asarray(ret)
    print(result.shape)

    plt.imshow(result[0][2])
    plt.savefig('seg1.png')
    plt.clf()

    plt.imshow(result[1][2])
    plt.savefig('seg2.png')
    plt.clf()

    result = result.astype('float')
    result[result == 0] = np.nan

    plt.imshow(np.abs(img[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.imshow(result[0][2], cmap='jet', alpha=0.5, aspect='auto')
    plt.savefig('fullseg1.png')
    plt.clf()

    plt.imshow(np.abs(img2[0][2] / 1961.06), cmap="Greys_r", interpolation="nearest", aspect='auto')
    plt.imshow(result[1][2], cmap='jet', alpha=0.5, aspect='auto')
    plt.savefig('fullseg2.png')
    plt.clf()


    # print(ret)




