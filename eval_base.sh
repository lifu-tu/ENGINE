# --remove-bpe: for evaluation ( BLEU4 )
python generate.py \
    PATH_YOUR_OUTPUT/$1   --gen-subset train \
    --path $2  \
    --beam 5

