export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATA_DIR="./dataset/elmo"

CUDA_VISIBLE_DEVICES=1 python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<elmo>" \
  --initializer_token="doll" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_elmo2"
#   --push_to_hub