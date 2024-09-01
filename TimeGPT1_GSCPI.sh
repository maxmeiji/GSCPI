
seq_len=32
data=GSCPI_Dataset
data_path=gscpi_data.csv
data_type=test

python -u FM.py \
  --model_name TimeGPT1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id GSCPI_${seq_len}_6 \
  --data $data \
  --data_type $data_type \
  --seq_len $seq_len \
  --pred_len 6 \
  --batch_size 8 \
  --device gpu


