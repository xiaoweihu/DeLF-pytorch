
# CUDA_VISIBLE_DEVICES=5 python run.py \
#     --stage 'keypoint' \
#     --ncls 1093 \
#     --optim 'sgd' \
#     --use_random_gamma_rescale \
#     --expr 'brandsports_bneval' \
#     --load_from data/brand_output/brandsports_resnet50_nobg/snapshot/model_best.pth.tar \
#     --data_cfg_path data/brand_output/configs/train_nobg.yamllst \
#     --bn_eval_mode \
#     --train_batch_size 64 \
#     --val_batch_size 64 \
#     --num_epochs 100 \
#     --weight_decay 0.001 \
#     --lr-policy MULTIFIX \
