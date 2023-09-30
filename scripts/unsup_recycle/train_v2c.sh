# viper to cityscapes
python train_ShortcutV2V.py --gpu_ids 0 --dataroot $dataset_root \
--name ShortcutV2V_unsup_v2c --dataset_mode video --main_G_path checkpoints/v2c_teacher/latest_net_G_A.pth \
--feat_ch 128 --h_dim 64 --dataset_option v2c --skip_idx_start 5 --skip_idx_end 17 \
--batchSize 16 --niter_decay 200 --niter 3000 --save_latest_freq 2000 --tensorboard_dir tensorboard/ --max_interval 3