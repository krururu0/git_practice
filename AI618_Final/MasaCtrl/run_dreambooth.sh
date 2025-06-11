# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export INSTANCE_DIR="./dataset/messi"
# export OUTPUT_DIR="./dreambooth/dreambooth_messi3"
# export CLASS_DIR="./dreambooth/dreambooth_messi3/class"

# CUDA_VISIBLE_DEVICES=5 accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=2e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --checkpointing_steps=500 \
#   --with_prior_preservation \
#   --class_data_dir=$CLASS_DIR \
#   --class_prompt="a photo of a person" \
#   --validation_prompt="a photo of a sks person"\
#   --num_validation_images=4 \
#   --validation_steps=100

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="./dataset/messi"
export OUTPUT_DIR="./dreambooth/dreambooth_messi3"
export CLASS_DIR="./dreambooth/dreambooth_messi3/class"

CUDA_VISIBLE_DEVICES=5 accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=4000 \
  --checkpointing_steps=500 \
  --validation_prompt="a photo of a sks person" \
  --num_validation_images=4 \
  --validation_steps=100


# export INSTANCE_DIR="./dataset/rolnaldo"
# export OUTPUT_DIR="./dreambooth/dreambooth_ronaldo"
# export CLASS_DIR="./dreambooth/dreambooth_ronaldo/class"

# CUDA_VISIBLE_DEVICES=5 accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=2e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --checkpointing_steps=500 \
#   # --with_prior_preservation \
#   # --class_data_dir=$CLASS_DIR \
#   # --class_prompt="a photo of a person" \
#   --validation_prompt="a photo of a sks person" \
#   --num_validation_images=4 \
#   --validation_steps=100


# export INSTANCE_DIR="./dataset/elmo"
# export OUTPUT_DIR="./dreambooth/dreambooth_elmo3"
# export CLASS_DIR="./dreambooth/dreambooth_elmo3/class"

# CUDA_VISIBLE_DEVICES=5 accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=2e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --checkpointing_steps=500 \
#   # --with_prior_preservation \
#   # --class_data_dir=$CLASS_DIR \
#   # --class_prompt="a photo of a person" \
#   --validation_prompt="a photo of a sks person" \
#   --num_validation_images=4 \
#   --validation_steps=100


# export INSTANCE_DIR="./dataset/curry"
# export OUTPUT_DIR="./dreambooth/dreambooth_curry"
# export CLASS_DIR="./dreambooth/dreambooth_curry/class"

# CUDA_VISIBLE_DEVICES=5 accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=2e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=2000 \
#   --checkpointing_steps=500 \
#   # --with_prior_preservation \
#   # --class_data_dir=$CLASS_DIR \
#   # --class_prompt="a photo of a person" \
#   --validation_prompt="a photo of a sks person" \
#   --num_validation_images=4 \
#   --validation_steps=100