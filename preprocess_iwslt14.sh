## the raw text folder
text=data/iwslt14deenfair

## the beam search outputs(pseudocorpus)
#text=data/iwslt14deen_beam5fair

output_dir=PATH_YOUR_OUTPUT

src=de

tgt=en



####process the raw text
python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train --validpref $text/valid --testpref $text/test --destdir ${output_dir}/iwslt14deen --workers 60

####precoss the pseudocorpus
#python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train --validpref $text/valid --testpref $text/test  --srcdict  ENGINE_IWSLT14DEEN/dict.de.txt   --tgtdict  ENGINE_IWSLT14DEEN/dict.en.txt   --destdir ${output_dir}/iwslt14deen_beam5 --workers 60
