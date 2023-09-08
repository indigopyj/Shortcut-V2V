python test_ShortcutV2V.py \
--dataroot /home/nas4_dataset/vision/ \
--name ShortcutV2V_unsup_v2c --dataset_mode video --main_G_path checkpoints/v2c_teacher/latest_net_G_A.pth \
--how_many 2 --results_dir ./results/ --which_epoch latest --max_interval 3
