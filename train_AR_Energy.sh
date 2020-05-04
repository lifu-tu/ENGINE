CUDA_VISIBLE_DEVICES=0  python train.py \
    PATH_YOUR_OUTPUT/roen  \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --update-freq $1 \
    --lr $2 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --save-dir checkpoints_enro$1_$2 

