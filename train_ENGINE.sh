##### an example for training ENGINE
## the data source(process the script)
text=roen_beam5

### the pretrained CMLM
Pretrained_cmlm=CMLM_RO2EN.pt

### the folder for saving the training models
output_dir=RO2EN

####the pretrained autoregressive transformer model
energy=AR_Energy_RO2EN.pt


mkdir -p $output_dir; cp ENGINE_WMT16ROEN/$Pretrained_cmlm  $output_dir/checkpoint_last.pt; CUDA_VISIBLE_DEVICES=0  python train_inf.py   PATH_YOUR_OUTPUT/$text  --always-mask   --energy_file  ENGINE_WMT16ROEN/$energy   --infnet-toE-feed-type 0  --feed-type 0   --arch bert_transformer_seq2seq   --share-all-embeddings --update-freq 8   --reset-optimizer  --criterion  Inf_Energy_Loss  --label-smoothing 0.1 --lr 0.000001   --alpha 0   --optimizer adam --adam-betas '(0.9, 0.999)'  --task translation_inf --max-tokens 1024 --weight-decay 0.01 --dropout 0.1 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512   --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir  $output_dir
