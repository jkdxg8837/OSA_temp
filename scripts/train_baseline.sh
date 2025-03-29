python main_baseline.py --batch_size 256 --epochs 5 --lr 1e-5 --warmup 500 --experiments cc3m \
 --vision_model ViT-B/32 --dataset cc3m --dataset_root None \
 --checkpoint_path ${SAVE_PATH} --noise_ratio 0.0 --cache_dir /dcs/pg24/u5649209/data/OSA_temp/cache