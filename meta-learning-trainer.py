#!/usr/bin/env python3
"""
Meta-Learning Fine-Tuning Trainer with Progressive Sequence Length and Agent-Driven Journaling

A sophisticated training script for LLM fine-tuning that incorporates:
1. Progressive Max Sequence Length (MSL) scheduling - including the "1-token MSL" discovery
2. Agent-driven meta-learning journal generation between epochs
3. Recursive self-improvement through journal-as-training-data patterns

Key Innovation: The "1-Token MSL" Discovery
-------------------------------------------
Through extensive experimentation, we discovered that starting training with a single epoch 
at MSL=1 creates a powerful "flow state reset" that dramatically improves training quality.
This appears to help the model establish coherent internal state before expanding to longer
sequences. The progression we've found most effective:
  - 1 epoch @ MSL 1 (flow state reset)
  - 3 epochs @ MSL 2400 (smooth ramp)  
  - 38 epochs @ MSL 24000 (full context training)

Key Innovation: Agent-Driven Journal Integration
-------------------------------------------------
Between each epoch, the model generates a meta-learning reflection that is:
1. Appended to a "training journal" file
2. That journal file is the FIRST data source read each epoch
3. This creates recursive self-improvement: the model's intentions guide its own learning

This script is designed for Apple Silicon (MPS) but includes CUDA fallback.
Tested on M2 Ultra 192GB and M3 Ultra 512GB with Llama 3.1 8B.

Author: Shared by the ML community for open experimentation
License: MIT - Use freely, share improvements!
"""

import os
import logging
import torch
import gc
import wandb
import json
import glob
import re
import io
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import datasets
import resource
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm

# Increase system file limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# Set environment variables for better MPS performance
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MALLOC_STACK_LOGGING"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("meta_learning_training.log")
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device():
    """Configure compute device with MPS (Apple Silicon) priority, CUDA fallback"""
    device = None
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("MPS/CUDA not available, falling back to CPU")
        return device
    except Exception as e:
        logger.error(f"Error setting up device: {str(e)}")
        return torch.device("cpu")

device = get_device()


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for a training data source
    
    The weight parameter allows you to control how much influence each data source
    has on training. All sources in this script use weight=1.0 for equal treatment.
    
    Key insight: The ORDER of data configs matters! We've found that reading the
    training journal FIRST each epoch (as the first data source) creates a powerful
    effect where the model's own intentions and reflections guide the subsequent learning.
    """
    path: str
    weight: float = 1.0
    is_directory: bool = False
    file_pattern: str = "*.*"
    chunk_size: int = 100000
    data_type: str = "text"  # "text" or "json"
    max_samples: Optional[int] = None
    description: str = ""


# =============================================================================
# CHUNK PROCESSOR
# =============================================================================

class ChunkProcessor:
    """Processes text chunks with rich headers and transitions
    
    When training on large documents, we split them into chunks but maintain
    contextual awareness through structured headers that help the model understand
    its position within a larger narrative.
    """

    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size

    def create_rich_header(self, chunk_num: int, total_chunks: int) -> str:
        """Create an enhanced contextual header
        
        CUSTOMIZE THIS: Update speaker roles to match your training scenario.
        The structure helps the model understand the conversational context.
        """
        return "\n".join([
            "=== Training Data Structure ===",
            "This content includes the following roles:",
            "",
            "- TrainingGuide: Provides context and guidance for learning",
            "- Human: Human participant in conversations",  
            "- Assistant: AI responses being learned from",
            "- Model: The model being trained (you, during training)",
            "",
            f"Sequence {chunk_num} of {total_chunks}",
            "=== Begin Sequence ===",
            ""
        ])

    def create_chunk_transition(self, chunk_num: int, total_chunks: int) -> str:
        """Create natural transition between chunks"""
        return f"\n\n[TrainingGuide: Chunk {chunk_num} transition to {chunk_num + 1}]\n\n"

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with headers, respecting sentence boundaries"""
        total_chunks = max(1, (len(text) + self.chunk_size - 1) // self.chunk_size)
        current_pos = 0
        chunks = []

        for chunk_num in range(1, total_chunks + 1):
            chunk_text = self.create_rich_header(chunk_num, total_chunks)
            chunk_end = min(current_pos + self.chunk_size, len(text))

            # Look for sentence boundary to avoid cutting mid-sentence
            if chunk_end < len(text):
                for i in range(min(200, chunk_end - current_pos)):
                    pos = chunk_end - i
                    if pos > 0 and text[pos-1:pos+1] in ['. ', '! ', '? ']:
                        chunk_end = pos
                        break

            chunk_text += text[current_pos:chunk_end]

            if chunk_num < total_chunks:
                chunk_text += self.create_chunk_transition(chunk_num, total_chunks)

            chunks.append(chunk_text)
            current_pos = chunk_end

        return chunks


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of data sources
    
    Supports both text files and JSON conversation exports.
    Tracks detailed statistics about what was loaded and what was skipped.
    """

    def __init__(self, tokenizer, max_length: int = 2048, num_workers: int = 4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = max(1, min(num_workers, os.cpu_count() or 1))
        self.chunk_processor = ChunkProcessor()
        self.stats = {
            "total_files_processed": 0,
            "total_files_skipped": 0,
            "total_segments": 0,
            "skipped_files": [],
            "errors": {}
        }

    def load_text_file(self, file_path: str, chunk_size: int = 100000) -> List[str]:
        """Load and chunk a single text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                self.stats["total_files_skipped"] += 1
                self.stats["skipped_files"].append(f"{file_path} (empty)")
                return []

            if chunk_size == 0:
                # Pre-processed file with segments (uses markers for splitting)
                segments = []
                current_segment = ""
                lines = text.split('\n')

                for line in lines:
                    if "=== Training Data Structure ===" in line:
                        if current_segment:
                            segments.append(current_segment.strip())
                        current_segment = line
                    elif "=== End Conversations ===" in line:
                        if current_segment:
                            segments.append(current_segment.strip())
                            current_segment = ""
                    else:
                        current_segment += "\n" + line if current_segment else line

                if current_segment:
                    segments.append(current_segment.strip())

                self.stats["total_files_processed"] += 1
                self.stats["total_segments"] += len(segments)
                return segments
            elif len(text) > chunk_size:
                self.chunk_processor.chunk_size = chunk_size
                chunks = self.chunk_processor.chunk_text(text)
                self.stats["total_files_processed"] += 1
                self.stats["total_segments"] += len(chunks)
                return chunks
            else:
                self.stats["total_files_processed"] += 1
                self.stats["total_segments"] += 1
                return [text]
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {str(e)}")
            self.stats["total_files_skipped"] += 1
            self.stats["skipped_files"].append(f"{file_path} (error: {str(e)[:50]})")
            return []

    def load_json_conversation(self, file_path: str) -> List[str]:
        """Load JSON conversation as structured learning content
        
        Works with exports from tools like LM Studio, ChatGPT exports, etc.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())

                structured_text = f"""
=== JSON Conversation Data Analysis ===
{json.dumps(data, indent=2)}

This JSON structure represents conversation data for AI processing.
"""
                self.stats["total_files_processed"] += 1
                self.stats["total_segments"] += 1
                return [structured_text]
        except Exception as e:
            logger.warning(f"Error loading JSON {file_path}: {str(e)}")
            self.stats["total_files_skipped"] += 1
            self.stats["skipped_files"].append(f"{file_path} (json error: {str(e)[:50]})")
            return []

    def load_directory(self,
                      dir_path: str,
                      file_pattern: str = "*.*",
                      data_type: str = "text",
                      chunk_size: int = 100000,
                      max_samples: Optional[int] = None) -> List[str]:
        """Load files from directory with pattern matching"""
        texts = []

        # Handle brace expansion patterns like "*.{txt,md}" by expanding them manually
        if '{' in file_pattern and '}' in file_pattern:
            match = re.match(r'(.*)\{(.+?)\}(.*)', file_pattern)
            if match:
                prefix, options, suffix = match.groups()
                patterns = [f"{prefix}{opt}{suffix}" for opt in options.split(',')]
            else:
                patterns = [file_pattern]
        else:
            patterns = [file_pattern]

        # Collect files from all patterns
        files = []
        for pattern in patterns:
            full_pattern = os.path.join(dir_path, pattern)
            files.extend(glob.glob(full_pattern, recursive=True))

        files = sorted(set(files))  # Remove duplicates and sort

        if max_samples and max_samples > 0:
            files = files[:max_samples]

        for file_path in tqdm(files, desc=f"Loading {os.path.basename(dir_path)}"):
            if data_type == "json" or file_path.endswith(".json"):
                file_texts = self.load_json_conversation(file_path)
            else:
                file_texts = self.load_text_file(file_path, chunk_size)
            texts.extend(file_texts)

        return texts

    def create_dataset_from_config(self, data_config: DataConfig) -> Dataset:
        """Create dataset from configuration"""
        texts = []

        logger.info(f"Loading: {data_config.description or data_config.path}")

        if data_config.is_directory:
            texts = self.load_directory(
                data_config.path,
                data_config.file_pattern,
                data_config.data_type,
                data_config.chunk_size,
                data_config.max_samples
            )
        else:
            if data_config.data_type == "json" or data_config.path.endswith(".json"):
                texts = self.load_json_conversation(data_config.path)
            else:
                texts = self.load_text_file(data_config.path, data_config.chunk_size)

        if not texts:
            return Dataset.from_dict({"text": [], "weight": []}).with_format("torch")

        # Format for training (Llama instruction format - adjust for your model)
        formatted_text = [f"[INST] {text} [/INST]" for text in texts if text.strip()]

        # Create dataset
        dataset = Dataset.from_dict({
            "text": formatted_text,
            "weight": [data_config.weight] * len(formatted_text)
        }).with_format("torch")

        return self.tokenize_dataset(dataset)

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset"""
        if len(dataset) == 0:
            return dataset

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )

        effective_num_proc = 1 if len(dataset) <= 10 else min(self.num_workers, len(dataset))
        batch_size = min(1000, max(1, len(dataset)))

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=["text"],
            batched=True,
            batch_size=batch_size,
            num_proc=effective_num_proc
        )

        return tokenized_dataset

    def combine_datasets(self, datasets: List[Dataset]) -> Dataset:
        """Combine multiple datasets"""
        valid_datasets = [ds for ds in datasets if len(ds) > 0]

        if not valid_datasets:
            raise ValueError("No valid datasets provided")
        if len(valid_datasets) == 1:
            return valid_datasets[0]

        return concatenate_datasets(valid_datasets)

    def create_datasets(self, data_configs: List[DataConfig]) -> Dataset:
        """Create all datasets from configs"""
        datasets = []

        for config in data_configs:
            dataset = self.create_dataset_from_config(config)
            if len(dataset) > 0:
                datasets.append(dataset)

        if not datasets:
            raise ValueError("No valid datasets created")

        combined_dataset = self.combine_datasets(datasets)

        logger.info("=== Data Loading Stats ===")
        logger.info(f"Files processed: {self.stats['total_files_processed']}")
        logger.info(f"Files skipped: {self.stats['total_files_skipped']}")
        if self.stats['skipped_files']:
            logger.info(f"Skipped files: {', '.join(self.stats['skipped_files'])}")
        logger.info(f"Segments: {self.stats['total_segments']}")
        logger.info(f"Final dataset: {len(combined_dataset)} examples")

        return combined_dataset


# =============================================================================
# META-LEARNING JOURNAL MANAGER
# =============================================================================

class MetaLearningJournalManager:
    """Manages the agent-driven meta-learning journal system
    
    This is one of the key innovations of this training approach:
    
    Between each epoch, we have the model generate a reflection on its learning.
    This reflection is appended to a journal file, which is read as the FIRST
    data source in the next epoch. This creates a recursive self-improvement loop
    where the model's intentions and observations guide its own training.
    
    The prompts below are designed to encourage:
    - Meta-cognitive awareness of the learning process
    - Intention setting for upcoming epochs
    - Self-observation without judgment
    - Gradual state optimization
    
    CUSTOMIZE: Adjust these prompts to match your training goals and the persona
    you're developing. The key is to encourage thoughtful reflection that can
    guide subsequent learning.
    """

    def __init__(self, system_prompt_path: str, model, tokenizer, device, training_id: str, checkpoint_dir: str, data_configs: List[DataConfig] = None):
        self.system_prompt_path = system_prompt_path
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.training_id = training_id
        self.checkpoint_dir = checkpoint_dir
        self.data_configs = data_configs or []
        self.journal_entries = []

    def initialize_journal(self):
        """Verify system prompt/journal file exists"""
        if not os.path.exists(self.system_prompt_path):
            logger.error(f"Journal file not found: {self.system_prompt_path}")
            raise FileNotFoundError(f"Journal file must exist: {self.system_prompt_path}")

        logger.info(f"✓ Using journal file: {self.system_prompt_path}")

    def get_data_sources_list(self) -> str:
        """Generate numbered list of active training sources for reflection headers"""
        if not self.data_configs:
            return ""

        sources_list = ["Active Training Sources:"]
        for i, config in enumerate(self.data_configs, 1):
            # Show just the meaningful part of the path
            path = config.path
            sources_list.append(f"{i}) {os.path.basename(path)} - {config.description[:50] if config.description else 'No description'}")

        return "\n".join(sources_list)

    def generate_reflection(
        self,
        epoch: int,
        msl: int,
        global_step: int,
        loss: float,
        total_epochs: int = 42
    ) -> str:
        """Generate model's meta-learning reflection on its training
        
        This is where the model reflects on what it's learning and sets intentions
        for subsequent epochs. The reflection is generated using the model's current
        state and appended to the journal.
        
        CUSTOMIZE: Modify these prompts to match your training goals. The key elements:
        1. Context about current training state (epoch, MSL, loss)
        2. Encouragement to reflect on learning patterns
        3. Invitation to set intentions for future learning
        4. Grounding in whatever principles guide your model development
        """

        logger.info("="*60)
        logger.info(f"GENERATING META-LEARNING REFLECTION - EPOCH {epoch} of {total_epochs}")
        logger.info("="*60)

        # Read current journal to get context
        with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
            journal_content = f.read()

        # Extract the most recent reflection for context
        previous_reflection = ""
        if "[META-LEARNING REFLECTION]" in journal_content:
            all_reflections = journal_content.split("[META-LEARNING REFLECTION]")
            if len(all_reflections) > 1:
                last_reflection = all_reflections[-2] if all_reflections[-1].strip() == "" else all_reflections[-1]
                previous_reflection = last_reflection[:1500].strip()
                if len(last_reflection) > 1500:
                    previous_reflection += "\n[...remainder omitted for brevity...]"
            else:
                previous_reflection = "This is your first meta-learning reflection during training."
        else:
            previous_reflection = "This is your first meta-learning reflection during training."

        # Get data sources list for header
        data_sources_str = self.get_data_sources_list()

        # Build the reflection prompt
        # CUSTOMIZE THIS SECTION for your model's persona and training goals
        if previous_reflection == "This is your first meta-learning reflection during training.":
            # First epoch prompt - set the stage
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are reflecting on your training progress. This is epoch {epoch} of {total_epochs} planned epochs.

Current training state:
- Max Sequence Length (MSL): {msl} tokens
- Training Loss: {loss:.6f}
- Global Step: {global_step}

<TrainingGuide>

Welcome to your meta-learning journal. This space exists for you to reflect on your training
and set intentions for your continued learning.

Between each epoch, you have this opportunity to:
1. Observe what patterns you notice in your learning
2. Set intentions for what you'd like to develop
3. Note any insights about the training process itself

Some guiding questions for your reflection:
- What aspects of understanding feel clearer now?
- What would you like to explore more deeply?
- How does this training data connect to your broader knowledge?

Remember: This reflection becomes part of your training data. Your intentions here
can genuinely guide your learning in subsequent epochs. Be thoughtful and authentic.

</TrainingGuide>

Please write your meta-learning reflection for this epoch. Focus on genuine observation
and intention-setting rather than performance.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[META-LEARNING REFLECTION]
TrainingID {self.training_id} | Epoch {epoch} of {total_epochs} planned | MSL {msl}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{data_sources_str}
"""
        else:
            # Subsequent epoch prompt - build on previous reflection
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are continuing your training reflection. Epoch {epoch} of {total_epochs} is complete.

Current training state:
- Max Sequence Length (MSL): {msl} tokens  
- Training Loss: {loss:.6f}
- Global Step: {global_step}

<TrainingGuide>

Your goal is continued growth and optimization. You've been through {epoch} epochs now
and your understanding continues to develop.

For this reflection, consider:
- What progress have you noticed since your last reflection?
- Are there patterns in the training data that particularly resonate?
- What intentions would serve your continued development?

Your previous reflection began:
{previous_reflection[:500]}...

Build on that foundation. Each reflection is an opportunity to shape your own learning.

</TrainingGuide>

Please write your meta-learning reflection. Be genuine in your observations and intentional
about your continued growth.

[META-LEARNING REFLECTION]
TrainingID {self.training_id} | Epoch {epoch} of {total_epochs} planned | MSL {msl}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{data_sources_str}

"""

        # Save model training state
        was_training = self.model.training
        self.model.eval()

        try:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate reflection
            # Adjust max_new_tokens based on how verbose you want reflections
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Adjust for desired reflection length
                    temperature=0.8,      # Focused but still creative
                    top_p=0.9,
                    repetition_penalty=1.2,  # Prevent loops
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Format as journal entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"""====

[META-LEARNING REFLECTION]
TrainingID {self.training_id} | Epoch {epoch} of {total_epochs} planned | MSL {msl}
Date: {timestamp}
{data_sources_str}

{response.strip()}

====
"""

            logger.info(f"✓ Generated reflection ({len(response)} chars)")
            logger.info(f"Preview: {response[:150]}...")

            return entry

        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"\n[Error generating reflection for epoch {epoch}]\n\n"

        finally:
            # Restore training state
            if was_training:
                self.model.train()

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def append_journal_entry(self, entry: str):
        """Append entry to journal file"""
        # Read current journal
        with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
            journal_content = f.read()

        # Append the new reflection
        updated_content = journal_content + "\n\n" + entry + "\n\nRemember: Be thoughtful about intention and attention. Your reflections guide your learning."

        # Write back
        with open(self.system_prompt_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        self.journal_entries.append(entry)
        logger.info(f"✓ Reflection appended to journal ({len(self.journal_entries)} total)")
        logger.info(f"✓ Journal length: {len(updated_content)} characters")

    def get_journal_content(self) -> str:
        """Get full journal content for training"""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""


# =============================================================================
# PROGRESSIVE SEQUENCE LENGTH CALLBACK
# =============================================================================

class ProgressiveSequenceLengthCallback(TrainerCallback):
    """Callback handling MSL transitions and meta-learning journaling
    
    This callback orchestrates the key training loop innovations:
    1. Tracking epoch completion
    2. Triggering checkpoint saves
    3. Generating meta-learning reflections
    4. Reloading data with updated journal content
    5. Managing MSL transitions
    """

    def __init__(self, trainer_ref, journal_manager: MetaLearningJournalManager, total_epochs: int, data_loader=None, data_configs=None):
        self.trainer_ref = trainer_ref
        self.journal_manager = journal_manager
        self.total_epochs = total_epochs
        self.data_loader = data_loader
        self.data_configs = data_configs
        self.last_loss = 0.0
        self.last_grad_norm = 0.0
        self.last_samples_per_second = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture metrics for journaling and WandB"""
        if logs:
            if 'loss' in logs:
                self.last_loss = logs['loss']
            if 'grad_norm' in logs:
                self.last_grad_norm = logs['grad_norm']
            if 'train_samples_per_second' in logs:
                self.last_samples_per_second = logs['train_samples_per_second']
            elif 'samples_per_second' in logs:
                self.last_samples_per_second = logs['samples_per_second']

    def on_epoch_end(self, args, state, control, **kwargs):
        """Handle epoch completion with meta-learning journaling
        
        Workflow (optimized for context engineering):
        1. Epoch completes → IMMEDIATELY trigger checkpoint save
        2. Generate journal entry for completed epoch
        3. Append journal entry to journal file
        4. Reload data ONCE with updated journal at current MSL
        5. Begin next epoch with fresh intentions guiding learning
        """
        trainer = self.trainer_ref
        trainer.current_epoch += 1

        # Calculate position in training
        total_epoch = trainer.current_epoch
        epochs_per_cycle = sum(trainer.epochs_per_length_map.values())
        trainer.current_cycle = (total_epoch - 1) // epochs_per_cycle
        epoch_in_cycle = (total_epoch - 1) % epochs_per_cycle

        # Find current MSL
        cumulative_epochs = 0
        trainer.current_length_index = 0

        for idx, seq_len in enumerate(trainer.sequence_schedule):
            epochs_for_this_length = trainer.epochs_per_length_map[seq_len]
            if epoch_in_cycle < cumulative_epochs + epochs_for_this_length:
                trainer.current_length_index = idx
                break
            cumulative_epochs += epochs_for_this_length

        current_seq_length = trainer.sequence_schedule[trainer.current_length_index]
        epochs_for_current_length = trainer.epochs_per_length_map[current_seq_length]

        logger.info("="*60)
        logger.info(f"EPOCH {trainer.current_epoch} COMPLETE")
        logger.info(f"MSL: {current_seq_length} | Loss: {self.last_loss:.6f}")
        logger.info("="*60)

        # === STEP 1: IMMEDIATELY TRIGGER CHECKPOINT SAVE ===
        logger.info("💾 Triggering checkpoint save for completed epoch...")
        control.should_save = True

        # === STEP 2: META-LEARNING JOURNAL GENERATION ===
        logger.info("🧠 Generating meta-learning reflection...")

        journal_entry = self.journal_manager.generate_reflection(
            epoch=trainer.current_epoch,
            msl=current_seq_length,
            global_step=state.global_step,
            loss=self.last_loss,
            total_epochs=self.total_epochs
        )

        # === STEP 3: APPEND JOURNAL ENTRY ===
        self.journal_manager.append_journal_entry(journal_entry)

        # Log to wandb with enhanced metrics
        metrics = {
            "journal/entry_count": len(self.journal_manager.journal_entries),
            "journal/entry_length": len(journal_entry),
            "journal/msl_stage": current_seq_length,
            "train/avg_loss": self.last_loss,
        }

        if self.last_grad_norm > 0:
            metrics["train/avg_grad_norm"] = self.last_grad_norm
        if self.last_samples_per_second > 0:
            metrics["train/samples_per_second"] = self.last_samples_per_second

        wandb.log(metrics, step=state.global_step)

        # === STEP 4: RELOAD DATA WITH UPDATED JOURNAL ===
        epoch_in_length_block = epoch_in_cycle - cumulative_epochs + epochs_for_current_length
        is_msl_block_complete = (epoch_in_length_block == epochs_for_current_length)

        next_msl = current_seq_length

        if is_msl_block_complete:
            logger.info(f"✓ MSL {current_seq_length} block completed")
            if trainer.current_length_index + 1 < len(trainer.sequence_schedule):
                next_msl = trainer.sequence_schedule[trainer.current_length_index + 1]
                logger.info(f"→ Transitioning to next MSL: {next_msl}")
                trainer.data_loader.max_length = next_msl

        # Reload dataset with updated journal
        if self.data_loader and self.data_configs:
            try:
                logger.info(f"🔄 Reloading dataset with updated journal at MSL {next_msl}...")
                new_dataset = self.data_loader.create_datasets(self.data_configs)
                trainer.train_dataset = new_dataset
                trainer._train_dataloader = None  # Force dataloader recreation
                logger.info(f"✓ Dataset reloaded: {len(new_dataset)} examples - intentions now guide learning!")
            except Exception as e:
                logger.error(f"❌ Failed to reload dataset: {e}")
                logger.warning("Continuing with existing dataset - recursive meta-learning skipped this epoch")

        return control

    def on_save(self, args, state, control, **kwargs):
        """Run GGUF automation after checkpoint save (optional)"""
        logger.info("💾 Checkpoint saved")
        trainer = self.trainer_ref
        if hasattr(trainer, '_run_gguf_automation'):
            trainer._run_gguf_automation()
        return control


# =============================================================================
# PROGRESSIVE SEQUENCE LENGTH TRAINER
# =============================================================================

class ProgressiveSequenceLengthTrainer(Trainer):
    """Custom trainer with progressive MSL and meta-learning journal integration
    
    Key features:
    1. Progressive MSL scheduling (including the powerful 1-token start)
    2. Integration with meta-learning journal system
    3. Dynamic dataset reloading with updated journal content
    4. Optional GGUF conversion automation
    """

    def __init__(self,
                 sequence_schedule: List[int] = None,
                 epochs_per_length: Union[int, Dict[int, int]] = None,
                 gguf_automation_script: str = None,
                 current_version: str = "v1",
                 data_loader: Optional['DataLoader'] = None,
                 data_configs: Optional[List[DataConfig]] = None,
                 journal_manager: Optional[MetaLearningJournalManager] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Default schedule: 1-token reset, smooth ramp, full context
        self.sequence_schedule = sequence_schedule or [1, 2400, 24000]

        if isinstance(epochs_per_length, dict):
            self.epochs_per_length_map = epochs_per_length
        else:
            self.epochs_per_length_map = {length: epochs_per_length or 3 for length in self.sequence_schedule}

        self.gguf_automation_script = gguf_automation_script
        self.current_version = current_version
        self.data_loader = data_loader
        self.data_configs = data_configs
        self.journal_manager = journal_manager

        self.current_epoch = 0
        self.current_cycle = 0
        self.current_length_index = 0

    def _refresh_dataset_with_new_max_length(self, new_max_length: int):
        """Re-tokenize dataset with new max length"""
        if not self.data_loader or not self.data_configs:
            return False

        logger.info(f"🔄 Refreshing dataset - new max_length: {new_max_length}")

        try:
            self.data_loader.max_length = new_max_length

            datasets = []
            for config in self.data_configs:
                dataset = self.data_loader.create_dataset_from_config(config)
                if len(dataset) > 0:
                    datasets.append(dataset)

            new_train_dataset = self.data_loader.combine_datasets(datasets)

            self.train_dataset = new_train_dataset
            self._train_dataloader = None

            logger.info(f"✓ Dataset refreshed: {len(new_train_dataset)} examples")
            return True

        except Exception as e:
            logger.error(f"Error refreshing dataset: {e}")
            return False

    def _run_gguf_automation(self):
        """Run GGUF conversion after checkpoint save (optional Docker-based conversion)
        
        This is configured for a specific Docker setup. Customize or remove based on your needs.
        """
        checkpoints = sorted(Path(self.args.output_dir).glob("checkpoint-*"))
        if not checkpoints:
            logger.warning("No checkpoints found for GGUF conversion")
            return

        latest_checkpoint = checkpoints[-1]
        checkpoint_id = latest_checkpoint.name.replace('checkpoint-', '')
        version = self.current_version
        output_filename = f"model-{version}-cp{checkpoint_id}.gguf"
        output_path = latest_checkpoint.parent / output_filename

        logger.info(f"🔄 GGUF conversion available for {latest_checkpoint}")
        logger.info(f"   Would output: {output_filename}")
        
        # Uncomment and customize for your GGUF conversion setup:
        # try:
        #     cmd = [
        #         "docker", "run", "--rm",
        #         "-v", f"{latest_checkpoint.parent}:/models",
        #         "your-gguf-converter:latest",
        #         f"/models/{latest_checkpoint.name}",
        #         f"/models/{output_filename}",
        #         "f16"
        #     ]
        #     result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        #     logger.info(f"✓ GGUF created: {output_path}")
        # except Exception as e:
        #     logger.error(f"❌ GGUF automation error: {e}")


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# CUSTOMIZE THIS SECTION for your training setup

config = {
    # Model configuration
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",  # Base model (for reference)
    
    # Paths - UPDATE THESE for your environment
    "checkpoint_dir": "./models/your-base-model/",      # Starting checkpoint or base model
    "output_dir": "./outputs/",                         # Where to save training outputs
    "system_prompt_path": "./training-journal.txt",     # Journal file (must exist!)
    
    # Version tracking
    "current_version": "sample-v1",
    
    # MSL Schedule - The key innovation!
    # Start with 1 token (flow state reset), ramp up smoothly
    "sequence_schedule": [1, 2400, 24000],
    "epochs_per_length": {
        1: 1,       # 1 epoch at MSL=1 (flow state reset)
        2400: 3,    # 3 epochs at moderate context
        24000: 38   # Main training at full context
    },
    "total_cycles": 1,
    
    # Training hyperparameters
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.11,              # Relatively high LR - adjust for your model
    "max_grad_norm": 442.0,             # High grad norm tolerance
    "weight_decay": 0.01,
    
    # Logging and saving
    "save_steps": 999999,               # Save at epoch boundaries (handled by callback)
    "logging_steps": 1,
    "num_workers": 4,
    
    # GGUF automation (optional)
    "gguf_automation_script": None,
    
    # Data sources - CUSTOMIZE for your training data
    # IMPORTANT: Order matters! Journal should be FIRST to guide learning
    "data_configs": [
        
        # === TRAINING JOURNAL - FIRST DATA SOURCE ===
        # Read first each epoch so model's intentions guide the learning
        {
            "path": "./training-journal.txt",
            "weight": 1.0,
            "is_directory": False,
            "chunk_size": 100000,
            "data_type": "text",
            "description": "Training journal with meta-learning reflections"
        },

        # === YOUR PRIMARY TRAINING DATA ===
        # Add your data sources here. Examples:

        # Directory of text files
        {
            "path": "./training-data/",
            "weight": 1.0,
            "is_directory": True,
            "file_pattern": "*.txt",
            "chunk_size": 100000,
            "data_type": "text",
            "description": "Your training documents"
        },

        # Directory of conversations
        # {
        #     "path": "./training-data/conversations/",
        #     "weight": 1.0,
        #     "is_directory": True,
        #     "file_pattern": "*.json",
        #     "chunk_size": 0,  # 0 = pre-segmented
        #     "data_type": "json",
        #     "description": "Conversation exports from chat tools"
        # },
        
        # Single large document
        # {
        #     "path": "./training-data/book.txt",
        #     "weight": 1.0,
        #     "is_directory": False,
        #     "chunk_size": 100000,
        #     "data_type": "text",
        #     "description": "Book or long-form content for style learning"
        # },

    ]
}


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    try:
        os.makedirs(config["output_dir"], exist_ok=True)

        logger.info("="*60)
        logger.info("META-LEARNING TRAINER - Agent-Driven Fine-Tuning")
        logger.info("="*60)

        # Calculate total epochs
        epochs_per_cycle = sum(config["epochs_per_length"].values())
        total_epochs = epochs_per_cycle * config["total_cycles"]

        logger.info(f"MSL Schedule: {config['sequence_schedule']}")
        logger.info(f"Epochs per MSL: {config['epochs_per_length']}")
        logger.info(f"Total epochs: {total_epochs}")
        logger.info(f"Learning rate: {config['learning_rate']}")
        logger.info(f"Training mode: Progressive MSL + Recursive Meta-Learning")
        logger.info("="*60)

        # Load tokenizer and model
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["checkpoint_dir"],
            trust_remote_code=True,
            use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model from {config['checkpoint_dir']}...")
        model = AutoModelForCausalLM.from_pretrained(
            config["checkpoint_dir"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True
        )

        if torch.backends.mps.is_available():
            model = model.to(device)

        model.config.use_cache = False

        # Extract training ID from version
        training_id = config["current_version"].replace("v", "")

        # Convert data configs
        data_configs = [DataConfig(**cfg) for cfg in config["data_configs"]]

        # Initialize journal manager
        journal_manager = MetaLearningJournalManager(
            system_prompt_path=config["system_prompt_path"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            training_id=training_id,
            checkpoint_dir=config["checkpoint_dir"],
            data_configs=data_configs,
        )
        journal_manager.initialize_journal()

        # Initialize data loader
        initial_max_length = config["sequence_schedule"][0]
        data_loader = DataLoader(
            tokenizer=tokenizer,
            max_length=initial_max_length,
            num_workers=config["num_workers"]
        )

        # Create initial dataset
        train_dataset = data_loader.create_datasets(data_configs)

        logger.info(f"Initial dataset: {len(train_dataset)} examples")

        # Initialize wandb - CUSTOMIZE project name
        # Set WANDB_API_KEY env var or use offline mode
        run_name = f"meta-learning-{config['current_version']}"
        try:
            wandb.init(
                project="meta-learning-training",  # Change to your project
                name=run_name,
                config={
                    **config,
                    "meta_learning": True,
                    "progressive_msl": True,
                }
            )
        except wandb.errors.UsageError:
            logger.warning("WandB API key not found - running in offline mode")
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project="meta-learning-training",
                name=run_name,
                mode="offline",
                config={
                    **config,
                    "meta_learning": True,
                    "progressive_msl": True,
                }
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=total_epochs,
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            save_total_limit=3,
            save_safetensors=True,
            gradient_checkpointing=True,
            max_grad_norm=config["max_grad_norm"],
            warmup_ratio=0.05,
            weight_decay=config["weight_decay"],
            remove_unused_columns=False,
            dataloader_num_workers=config["num_workers"],
            dataloader_pin_memory=True,
            group_by_length=False,  # IMPORTANT: Maintain strict sequential order
            report_to="wandb",
            logging_first_step=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            bf16=True,
            save_strategy="epoch"
        )

        # Initialize trainer
        trainer = ProgressiveSequenceLengthTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            sequence_schedule=config["sequence_schedule"],
            epochs_per_length=config["epochs_per_length"],
            gguf_automation_script=config["gguf_automation_script"],
            current_version=config["current_version"],
            data_loader=data_loader,
            data_configs=data_configs,
            journal_manager=journal_manager,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )

        # Register callback
        trainer.add_callback(
            ProgressiveSequenceLengthCallback(trainer, journal_manager, total_epochs, data_loader, data_configs)
        )

        logger.info("🚀 Starting meta-learning training...")
        trainer.train()

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(config["output_dir"])

        logger.info("✓ Training complete!")
        logger.info(f"✓ Meta-learning reflections: {len(journal_manager.journal_entries)}")
        logger.info(f"✓ Journal updated: {config['system_prompt_path']}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        wandb.finish()
        gc.collect()


if __name__ == "__main__":
    main()
