
## the raw text folder
text=data/wmt16fair/ro-en-original

output_dir=PATH_YOUR_OUTPUT

src=ro

tgt=en


python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train.bpe --validpref $text/dev.bpe --testpref $text/test.bpe --destdir ${output_dir}/ro2en --workers 60 --srcdict ENGINE_WMT16ROEN/dict.${src}.txt --tgtdict ENGINE_WMT16ROEN/dict.${tgt}.txt
