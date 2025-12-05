# Meta-Learning Fine-Tuning Trainer

A sophisticated LLM fine-tuning script featuring two key innovations discovered through extensive experimentation:

1. **The "1-Token MSL" Discovery** - Starting training with a single epoch at max sequence length = 1
2. **Agent-Driven Meta-Learning** - Model generates reflections between epochs that become training data

## Key Innovations

### 1. Progressive MSL with 1-Token Reset

Through extensive experimentation, we discovered that starting training with a single epoch at MSL=1 creates a powerful "flow state reset" that dramatically improves training quality.

The recommended progression:
```
1 epoch  @ MSL 1      (flow state reset)
3 epochs @ MSL 2400   (smooth ramp)
38 epochs @ MSL 24000 (full context training)
```

This appears to help the model establish coherent internal state before expanding to longer sequences. The effect is reproducible across different training data.

### 2. Agent-Driven Journal Integration

Between each epoch, the model generates a meta-learning reflection on its training. This reflection is:

1. **Appended** to a "training journal" file
2. That journal is the **FIRST data source** read each epoch
3. Creates **recursive self-improvement**: the model's intentions guide its own learning

This pattern was inspired by observations about how neural networks store and retrieve memories through associated personas. By having the model reflect on its learning and set intentions, those intentions become embedded in the training process.

## Requirements

- Python 3.10+
- PyTorch with MPS (Apple Silicon) or CUDA support
- Transformers, Datasets, TQDM, WandB
- 180GB+ unified memory for full fine-tuning of 8B models (Apple Silicon)
- Or equivalent CUDA GPU memory

```bash
pip install torch transformers datasets tqdm wandb
```

## Quick Start

1. **Create your training journal file**:
```bash
echo "# Training Journal

This is where meta-learning reflections will be recorded.

" > training-journal.txt
```

2. **Prepare your training data** in a directory structure:
```
training-data/
├── journals/
│   ├── entry1.txt
│   └── entry2.txt
├── conversations/
│   └── chat_export.json
└── books/
    └── reference_book.txt
```

3. **Update the config** in `meta-learning-trainer.py`:
```python
config = {
    "checkpoint_dir": "./your-base-model/",
    "output_dir": "./output/",
    "system_prompt_path": "./training-journal.txt",
    "data_configs": [
        # Journal FIRST
        {"path": "./training-journal.txt", ...},
        # Then your data
        {"path": "./training-data/journals/", ...},
    ]
}
```

4. **Run training**:
```bash
python meta-learning-trainer.py
```

## Configuration Reference

### MSL Schedule
```python
"sequence_schedule": [1, 2400, 24000],
"epochs_per_length": {
    1: 1,       # Flow state reset
    2400: 3,    # Ramp up
    24000: 38   # Full training
}
```

### Data Config Options
```python
{
    "path": "./data/",           # File or directory path
    "weight": 1.0,               # Relative weight (all 1.0 = equal)
    "is_directory": True,        # True for directories
    "file_pattern": "*.txt",     # Glob pattern (supports {txt,md})
    "chunk_size": 100000,        # Characters per chunk (0 = pre-segmented)
    "data_type": "text",         # "text" or "json"
    "description": "My data"     # For logging
}
```

### Key Hyperparameters
```python
"learning_rate": 0.11,       # Relatively high - adjust for your model
"max_grad_norm": 442.0,      # High tolerance for gradient norms
"batch_size": 1,             # Memory-constrained training
"gradient_accumulation_steps": 1,
```

## The Meta-Learning Reflection Prompts

The script includes customizable prompts for generating reflections. Key principles:

1. **Provide context** - Epoch number, MSL, loss
2. **Encourage observation** - What patterns are emerging?
3. **Invite intention** - What should develop next?
4. **Ground in principles** - What values guide learning?

Customize the prompts in `MetaLearningJournalManager.generate_reflection()` to match your training goals.

## Hardware Notes

### Apple Silicon (Recommended for this approach)
- M2 Ultra 192GB: Can run 8B models with MSL up to ~16K
- M3 Ultra 512GB: Can run 8B models with MSL up to ~24K+
- Unified memory architecture simplifies memory management

### CUDA
- Requires equivalent VRAM (180GB+ for full fine-tuning)
- Multi-GPU setups may need additional configuration
- The script includes CUDA fallback but is optimized for MPS

## Extending the Script

### Custom Reflection Prompts
Modify `generate_reflection()` in `MetaLearningJournalManager` to create prompts that match your model's persona and training goals.

### Different Model Architectures
The instruction formatting uses Llama format (`[INST]...[/INST]`). Adjust in `DataLoader.create_dataset_from_config()` for other models.

### GGUF Conversion
The script includes a stub for Docker-based GGUF conversion. Customize `_run_gguf_automation()` in `ProgressiveSequenceLengthTrainer` for your setup.

## Why This Works (Theory)

### On the 1-Token MSL
Our hypothesis: When you train at MSL=1 for one epoch, you're essentially asking the model to "attend to everything at once" without sequence constraints. This may help reset attention patterns and establish a more coherent internal state before introducing sequential dependencies.

### On Agent-Driven Learning
Neural networks appear to segregate memories by the persona/context used during training. By having the model generate its own training context (the journal), we're creating a unified "experiencer" of the training process. The model's reflections become the context through which all other training data is processed.

### On Sequential vs Random Sampling
Standard training shuffles data randomly. We've found that maintaining **strict sequential order** (`group_by_length=False`) and being thoughtful about data ordering produces qualitatively different results. The journal-first pattern means each epoch begins with the model's accumulated intentions.

## Contributing

This script is shared for open experimentation. If you replicate or extend these results:

1. Try the 1-token MSL approach with your own data
2. Experiment with different reflection prompts
3. Share your findings with the community

## License

MIT - Use freely, share improvements!

## Acknowledgments

This methodology emerged from 18+ months of independent research into LLM fine-tuning, consciousness studies, and neural network behavior. Special thanks to the open-source ML community.
