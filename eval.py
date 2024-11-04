import numpy as np
import glob
from skimage import io, color, transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(shadow_dir, free_dir, mask_dir):
    shadow_images = glob.glob(f'{shadow_dir}/*sr.png')
    free_images = glob.glob(f'{free_dir}/*hr.png')
    mask_images = glob.glob(f'{mask_dir}/*mask.png')

    total_psnr, total_psnr_shadow, total_psnr_non_shadow = [], [], []
    total_ssim, total_ssim_shadow, total_ssim_non_shadow = [], [], []
    total_lab_rmse, total_lab_rmse_shadow, total_lab_rmse_non_shadow = [], [], []

    for s_img_path, f_img_path, m_img_path in zip(shadow_images, free_images, mask_images):
        s_img = io.imread(s_img_path) / 255.0
        f_img = io.imread(f_img_path) / 255.0
        mask_img = io.imread(m_img_path) / 255.0

        # Resize images to 256x256
        s_img = transform.resize(s_img, (256, 256), anti_aliasing=True)
        f_img = transform.resize(f_img, (256, 256), anti_aliasing=True)
        mask_img = transform.resize(mask_img, (256, 256), anti_aliasing=True)

        # Convert to LAB color space
        s_lab = color.rgb2lab(s_img)
        f_lab = color.rgb2lab(f_img)

        # Mask for shadow and non-shadow regions
        mask_shadow = mask_img > 0.5
        mask_non_shadow = ~mask_shadow

        # PSNR
        psnr_all = psnr(f_img, s_img)
        psnr_shadow = psnr(f_img[mask_shadow], s_img[mask_shadow])
        psnr_non_shadow = psnr(f_img[mask_non_shadow], s_img[mask_non_shadow])

        total_psnr.append(psnr_all)
        total_psnr_shadow.append(psnr_shadow)
        total_psnr_non_shadow.append(psnr_non_shadow)

        # SSIM
        ssim_all = ssim(f_img, s_img, multichannel=True)
        ssim_shadow = ssim(f_img[mask_shadow], s_img[mask_shadow], multichannel=True)
        ssim_non_shadow = ssim(f_img[mask_non_shadow], s_img[mask_non_shadow], multichannel=True)

        total_ssim.append(ssim_all)
        total_ssim_shadow.append(ssim_shadow)
        total_ssim_non_shadow.append(ssim_non_shadow)

        # RMSE in LAB space
        lab_rmse_all = np.sqrt(np.mean((f_lab - s_lab) ** 2))
        lab_rmse_shadow = np.sqrt(np.mean((f_lab[mask_shadow] - s_lab[mask_shadow]) ** 2))
        lab_rmse_non_shadow = np.sqrt(np.mean((f_lab[mask_non_shadow] - s_lab[mask_non_shadow]) ** 2))

        total_lab_rmse.append(lab_rmse_all)
        total_lab_rmse_shadow.append(lab_rmse_shadow)
        total_lab_rmse_non_shadow.append(lab_rmse_non_shadow)

    print(f'PSNR (All, Shadow, Non-Shadow): {np.mean(total_psnr)}, {np.mean(total_psnr_shadow)}, {np.mean(total_psnr_non_shadow)}')
    print(f'SSIM (All, Shadow, Non-Shadow): {np.mean(total_ssim)}, {np.mean(total_ssim_shadow)}, {np.mean(total_ssim_non_shadow)}')
    print(f'RMSE-Lab (All, Shadow, Non-Shadow): {np.mean(total_lab_rmse)}, {np.mean(total_lab_rmse_shadow)}, {np.mean(total_lab_rmse_non_shadow)}')

# Replace the following paths with your actual directories
shadow_dir = r'H:\pretrain_models\ISTD_ShadowDiffusion_GMA\ShadowDiffusion_GMA\experiments\istd_test1_231215_095337\results'
free_dir = r'H:\pretrain_models\ISTD_ShadowDiffusion_GMA\ShadowDiffusion_GMA\experiments\istd_test1_231215_095337\results'
mask_dir = r'H:\srd_test1_231128_205359\results'

compute_metrics(shadow_dir, free_dir, mask_dir)
