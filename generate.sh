python generate_cmlm.py PATH_YOUR_OUTPUT/$1 --path $2 --task translation_self  --remove-bpe   --max-sentences 20 --decoding-iterations 1  --decoding-strategy mask_predict --length-beam $3
