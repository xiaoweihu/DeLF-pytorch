
# if ps -p 1862 > /dev/null
# then
#    echo "1862 is running"
#    # Do something knowing the pid exists, i.e. the process with $PID is running
# fi

# CUDA_VISIBLE_DEVICES=1 python run.py \
#     --stage 'keypoint' \
#     --ncls 1093 \
#     --optim 'sgd' \
#     --use_random_gamma_rescale \
#     --expr 'delf_brandsports_bneval_wd0.0001_layer4' \
#     --load_from data/brand_output/brandsports_resnet50_nobg/snapshot/model_best.pth.tar \
#     --data_cfg_path data/brand_output/configs/train_nobg.yamllst \
#     --bn_eval_mode \
#     --train_batch_size 32 \
#     --val_batch_size 32 \
#     --num_epochs 100 \
#     --target_layer layer4 \
#     --weight_decay 0.0001 \
#     --lr-policy MULTIFIX \

CUDA_VISIBLE_DEVICES=0 python run.py \
    --stage 'keypoint' \
    --ncls 1093 \
    --optim 'sgd' \
    --use_random_gamma_rescale \
    --expr 'delf_brandsports_bneval_wd0.0001' \
    --load_from data/brand_output/brandsports_resnet50_nobg/snapshot/model_best.pth.tar \
    --data_cfg_path data/brand_output/configs/train_nobg.yamllst \
    --bn_eval_mode \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --num_epochs 100 \
    --weight_decay 0.0001 \
    --lr-policy MULTIFIX \

# CUDA_VISIBLE_DEVICES=2 python run.py \
#     --stage 'keypoint' \
#     --ncls 1093 \
#     --optim 'sgd' \
#     --use_random_gamma_rescale \
#     --expr 'delf_brandsports_bneval_wd0.0001_steplr' \
#     --load_from data/brand_output/brandsports_resnet50_nobg/snapshot/model_best.pth.tar \
#     --data_cfg_path data/brand_output/configs/train_nobg.yamllst \
#     --bn_eval_mode \
#     --train_batch_size 32 \
#     --val_batch_size 32 \
#     --num_epochs 120 \
#     --weight_decay 0.0001 \
#     --lr-policy STEP \
#     --lr_stepsize 30 \
#     --lr 0.01 \
