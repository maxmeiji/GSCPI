
seq_len=32
model_name=DLinear

python -u Baseline.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path gscpi_data.csv \
  --model_id GSCPI_$seq_len'_'6 \
  --model $model_name \
  --data custom \
  --features S \
  --target Rate \
  --seq_len $seq_len \
  --pred_len 6 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 

