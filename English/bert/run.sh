for function in typo knowledge cluster all
do
    python attack.py \
    --function $function \
    --const 10 \
    --confidence 0 \
    --lr 0.15 \
    --load fever/bert-fever-uncased.pth \
    --test-model fever/bert-fever-uncased.pth \
    --test-data fever/test-data-cooked.pkl
done