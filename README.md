# llm-from-scratch

## Project Description
This project implements a lightweight optimized LLM "from scratch" with the following capabilities:
- Hypernetwork for on-the-fly weight generation
- Student model (6 layers, d_model=512) with LoRA adapters
- Knowledge Distillation (from hypermodel)
- Pruning (up to 80% weight removal via MagnitudePruner or FisherPruner)
- Quantization (4-bit post-training quantization)
- RAG (Faiss IVFPQ) for external memory
- BPE + PQ tokenizer (vocab_size=4096, m=4, ks=256)

The resulting model weighs ≤50 MB and can run inference even on older CPUs.

---

## Project Structure
```
llm-from-scratch/
├── configs/
│   ├── advanced.yml
│   ├── default.yaml
│   ├── hyperparameter_search.yaml
│   └── small.yml
├── data/
│   ├── my_dataset_clean.jsonl
│   └── dataset.py
├── hypernetwork/
│   └── hypernet.py
├── student/
│   ├── model_student.py
│   ├── distill.py
│   └── DistillationLoss.py
├── pruning_quant/
│   ├── pruning.py
│   └── quantization.py
├── rag_memory/
│   ├── memory_bank.py
│   └── build_index.py
├── tokenizer/
│   ├── bpe_pq_tokenizer.py
│   └── simple_tokenizer.py
├── trainer/
│   └── train_advanced.py
├── utils/
│   ├── helpers.py
│   └── logger.py
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_distillation.py
│   ├── test_quantization.py
│   ├── test_pruning.py
│   ├── test_generation.py
│   ├── test_trainer.py
│   └── test_student.py
├── .flake8
├── .editorconfig
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── generate.py
├── train.py
├── main.py
├── search_hyperparameters.py
└── README.md
```

---

## Installation
```bash
git clone https://github.com/your_username/llm-from-scratch.git
cd llm-from-scratch

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt

# (Optional) dev dependencies for testing and linting
pip install -r requirements-dev.txt
pre-commit install
```

---

## Configuration

The main configuration file is `configs/advanced.yml`. Example:

```yaml
use_hypernet: true

hypernet:
  hidden_sizes: [32, 32]
  activation: relu

student:
  n_layers: 6
  d_model: 512
  n_heads: 8
  distill_alpha: 0.5
  use_lora: true
  lora_rank: 4
  ffn_rank: 2048
  dropout: 0.1
  gradient_checkpointing: true
  mixed_precision: true
  max_seq_len: 256

pruning:
  prune_rate: 0.8

quantization:
  bits: 4

tokenizer:
  vocab_size: 4096
  pq_m: 4
  pq_ks: 256

rag:
  enabled: true
  dim: 512
  nlist: 100
  m: 8
  nbits: 8

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  epochs: 10
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 500

generation:
  max_length: 256
  temperature: 1.0
  top_k: 50
  top_p: 0.9
  num_beams: 5
  early_stopping: true

seed: 42
device: cuda  # or cpu
```

---

## Training

1. **Build Faiss Index (RAG)**

   ```bash
   python rag_memory/build_index.py \
     --input data/my_dataset_clean.jsonl \
     --output rag_memory/index.faiss
   ```

2. **Pretrain HyperNetwork (optional)**

   ```bash
   python trainer/train_advanced.py \
     --config configs/advanced.yml \
     --pretrain_hypernet \
     --max_steps 100
   ```

3. **Train Student Model with Distillation + Pruning + Quantization**

   ```bash
   python trainer/train_advanced.py \
     --config configs/advanced.yml \
     --train_student \
     --apply_pruning \
     --quantize
   ```

   * `--train_student` — student model distillation
   * `--apply_pruning` — pruning after each epoch
   * `--quantize` — 4-bit quantization after training

4. **Logs and Checkpoints**

   * Logs in `output/logs` (wandb, tensorboard)
   * Checkpoints in `output/checkpoints`
   * Best model: `output/model_best.pth`
   * Quantized model: `output/model_quant.pth`

5. **Validation**
   Automatic validation during training, metrics are logged to wandb and tensorboard.

---

## Generation

```bash
python generate.py \
  --model_dir ./output \
  --config ./configs/advanced.yml \
  --prompt "Hello, how are you?" \
  --rag_index rag_memory/index.faiss \
  --max_length 128 \
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.9 \
  --num_beams 5 \
  --early_stopping \
  --use_fp16 \
  --batch_size 8 \
  --seed 42 \
  --output_format text \
  --output_path ./generated.txt
```

Example of multi-prompt mode:

```bash
python generate.py \
  --model_dir ./output \
  --config ./configs/advanced.yml \
  --prompts_file ./prompts.txt \
  --rag_index rag_memory/index.faiss \
  --batch_size 16 \
  --use_fp16 \
  --output_format json \
  --output_path ./results.json
```

---

## Testing

```bash
python -m pytest -v
```

Or individually:

```bash
python -m pytest tests/test_student.py -v
```

---

## DevOps and CI/CD

* **Dockerfile**: build image with all dependencies
* **docker-compose.yml**: example service deployment
* **GitHub Actions** for:

  * running tests on each PR
  * linting (flake8, pre-commit)
  * building Docker image on merge to main

**Dependencies**:

* PyTorch ≥ 1.12
* tokenizers
* faiss-cpu
* numpy, tqdm, yaml
* wandb (optional)
* pytest, flake8 (dev)

---

## Recommendations

* Set a single **seed** for reproducibility
* In case of OOM, decrease `batch_size` or increase `gradient_accumulation_steps`
* Use DVC/MLflow for model and dataset versioning
* Enable profiling via `--profiling` for performance monitoring

---

## License

MIT License (see `LICENSE` file)

---

## Contact

Questions and suggestions — in GitHub issues or email: [your_email@example.com](mailto:your_email@example.com) 