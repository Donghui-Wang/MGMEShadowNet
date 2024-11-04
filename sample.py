import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
import shutil

# import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadow.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None
    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    print(opt['datasets'])
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            print(dataset_opt['data_len'])
            dataset_opt['data_len'] = -1
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    sample_sum = opt['datasets']['val']['data_len']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    best_psnr = 0.0
    best_ssim = 0.0
    save_status = False
    best_result_path = ""
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'], min_max=(0, 1))  # uint8
                        # mask = Metrics.tensor2img(visuals['mask'], min_max=(0, 1))  # uint8
                        #
                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img1(
                            fake_img, '{}/{}_{}_mask.png'.format(result_path, current_step, idx))

                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)
                        avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)
                        # Metrics.save_img1(
                        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                    print(avg_psnr)
                    # 计算整个测试集的平均PSNR和SSIM
                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx

                    # 将平均PSNR和SSIM用作保存最佳模型的标准
                    if avg_psnr >= best_psnr and avg_ssim >= best_ssim:
                        best_psnr = avg_psnr
                        best_ssim = avg_ssim
                        save_status = True
                        # 设置结果路径，仅保留到 'results' 目录
                        best_result_path = result_path

                    r_path = os.path.dirname(result_path)  # 截取到 "experiments\\SRD_MaskGuide_Mask_240920_144638\\results"
                    print(r_path)
                    with os.scandir(r_path) as entries:
                        subfolders = [entry.name for entry in entries if entry.is_dir()]

                    for dir_name in subfolders:
                        # 获取最后一部分
                        best_last_part = os.path.basename(best_result_path)
                        print("best_last_part",best_last_part)  # 输出: 1
                        print(dir_name)
                        dir_path = os.path.join(r_path, dir_name)
                        print("dir_path", dir_path)
                        print("best_result_path", best_result_path)
                        print("dir_path != best_result_path", int(best_last_part) != int(dir_name))
                        if int(best_last_part) != int(dir_name):
                            if os.path.isdir(dir_path):
                                try:
                                    shutil.rmtree(dir_path)
                                    logger.info(f'Removed old directory: {dir_path}')
                                except Exception as e:
                                    logger.error(f'Failed to remove old directory {dir_path}: {e}')


                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    if avg_psnr >= best_psnr and best_ssim >= avg_ssim:
                        best_psnr = avg_psnr
                        best_ssim = avg_ssim
                        save_status = True
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0 and save_status:
                    logger.info('Saving models and training states.')
                    # 删除之前保存的模型，只保留最新的最佳模型
                    checkpoint_dir = os.path.join(opt['path']['checkpoint'])

                    for file_name in os.listdir(checkpoint_dir):
                        if not file_name.endswith(f'epoch_{current_epoch}_iter_{current_step}.pth'):
                            file_path = os.path.join(checkpoint_dir, file_name)
                            try:
                                os.remove(file_path)
                                logger.info(f'Removed old checkpoint: {file_path}')
                            except Exception as e:
                                logger.error(f'Failed to remove old checkpoint {file_path}: {e}')

                    diffusion.save_network(current_epoch, current_step)
                    save_status = False

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                if wandb_logger:
                    wandb_logger.log_metrics({'epoch': current_epoch - 1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        sample_imgs = []
        for idx in range(sample_sum):
            idx += 1
            diffusion.sample(continous=True)
            visuals = diffusion.get_current_visuals(sample=True)

            show_img_mode = 'grid'
            if show_img_mode == 'single':
                # single img series
                sample_img = visuals['SAM']  # uint8
                sample_num = sample_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sample_img[iter]),
                        '{}/{}_{}_sample_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sample_img = Metrics.tensor2img(visuals['SAM'])  # uint8
                Metrics.save_img(
                    sample_img, '{}/{}_{}_sample_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SAM'][-1]),
                    '{}/{}_{}_sample.png'.format(result_path, current_step, idx))

            sample_imgs.append(Metrics.tensor2img(visuals['SAM'][-1]))

        if wandb_logger:
            wandb_logger.log_images('eval_images', sample_imgs)
