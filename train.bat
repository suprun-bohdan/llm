@echo off
python train.py --config configs/train_config.json --data my_dataset_clean.jsonl --output_dir outputs/student_model
pause