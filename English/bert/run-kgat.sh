for function in typo knowledge cluster all
do
    CUDA_VISIBLE_DEVICES=3 python attack.py \
    --function $function \
    --const 10 \
    --confidence 0 \
    --lr 0.15 \
    --bert_pretrain /home/tuh17884/codes/KernelGAT/bert_base \
    --model-state /home/tuh17884/codes/KernelGAT/checkpoint/kgat/model.best.pt \
    --test-data fever/kgat-attack-data.pkl \
    --attack-model kgat \
    --sample 100 \
    --batch-size 1
done