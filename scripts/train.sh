NOISE_RATIO=0.0
DATASETS="cc3m"
SAVE_PATH="./outputs"
python main_clip.py --batch_size 256 --epochs 5 --lr 1e-5 --warmup 500 --experiments cc3m \
 --vision_model ViT-B/32 --dataset cc3m --dataset_root None \
 --checkpoint_path ${SAVE_PATH} --noise_ratio 0.0