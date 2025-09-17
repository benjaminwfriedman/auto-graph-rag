"""Fine-tune language models for Cypher generation."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import DatasetDict
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class SampleGenerationCallback(TrainerCallback):
    """Custom callback to generate sample outputs during training."""
    
    def __init__(self, tokenizer, sample_prompts, eval_steps=100):
        """Initialize the callback.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            sample_prompts: List of sample prompts to test
            eval_steps: Generate samples every N steps
        """
        self.tokenizer = tokenizer
        self.sample_prompts = sample_prompts
        self.eval_steps = eval_steps
        self.generated_samples = []
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self._generate_samples(model, state.global_step, state.epoch)
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch."""
        self._generate_samples(model, state.global_step, state.epoch)
    
    def _generate_samples(self, model, step, epoch):
        """Generate sample outputs."""
        if model is None:
            return
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating samples at step {step} (epoch {epoch:.2f})")
        logger.info(f"{'='*60}")
        
        model.eval()
        device = next(model.parameters()).device
        
        for i, prompt in enumerate(self.sample_prompts[:3], 1):  # Limit to 3 samples
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Log the sample
            logger.info(f"\nSample {i}:")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated}")
            
            # Store for later analysis
            self.generated_samples.append({
                "step": step,
                "epoch": epoch,
                "prompt": prompt,
                "generated": generated
            })
        
        model.train()
        logger.info(f"{'='*60}\n")
    
    def get_samples_history(self):
        """Return the history of generated samples."""
        return self.generated_samples


class LossPlottingCallback(TrainerCallback):
    """Custom callback to track and plot training losses."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called whenever logging occurs."""
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(state.global_step)
            
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
    
    def plot_losses(self, save_path: Optional[str] = None):
        """Plot the training and validation losses."""
        # Check if we have any data to plot
        if not self.train_losses and not self.eval_losses:
            logger.warning("No loss data available to plot. This may happen with very short training runs.")
            return {
                "train_steps": [],
                "train_losses": [],
                "eval_steps": [],
                "eval_losses": []
            }
        
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if self.train_losses:
            plt.plot(self.steps, self.train_losses, 'b-', label=f'Training Loss ({len(self.train_losses)} points)', linewidth=2, marker='o')
            logger.info(f"Plotting {len(self.train_losses)} training loss points")
        
        # Plot validation loss
        if self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses, 'r-', label=f'Validation Loss ({len(self.eval_losses)} points)', linewidth=2, marker='s')
            logger.info(f"Plotting {len(self.eval_losses)} validation loss points")
        
        # Set labels and title
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some styling
        plt.tight_layout()
        
        # Add text if very few points
        if len(self.train_losses) <= 3:
            plt.text(0.02, 0.98, f"Note: Only {len(self.train_losses)} training points collected.\nTry longer training or more frequent logging.", 
                    transform=plt.gca().transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot if path provided
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Loss plot saved to {save_path}")
        
        # Don't show plot in automated environments, just save
        if save_path:
            plt.close()  # Close to free memory
        else:
            plt.show()
        
        # Also return the data for programmatic access
        return {
            "train_steps": self.steps,
            "train_losses": self.train_losses,
            "eval_steps": self.eval_steps,
            "eval_losses": self.eval_losses
        }


class FineTuner:
    """Fine-tune models for Cypher generation."""
    
    def __init__(
        self,
        # model_name: str = "meta-llama/Llama-3.2-1B",
        model_name:str = "TinyLlama/TinyLlama_v1.1_math_code",
        lora_rank: int = 16,
        use_4bit: bool = None,
        device_map: str = "auto"
    ):
        """Initialize fine-tuner.
        
        Args:
            model_name: Base model to fine-tune
            lora_rank: LoRA rank for adaptation
            use_4bit: Whether to use 4-bit quantization (auto-detect if None)
            device_map: Device mapping strategy
        """
        self.model_name = model_name
        self.lora_rank = lora_rank
        
        # Auto-detect if CUDA is available
        if use_4bit is None:
            self.use_4bit = torch.cuda.is_available()
        else:
            self.use_4bit = use_4bit
        
        # Use CPU if no CUDA
        if not torch.cuda.is_available():
            self.device_map = "cpu"
            self.use_4bit = False  # Can't use 4bit without CUDA
        else:
            self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.output_dir = None
        self.loss_callback = None
        self.sample_callback = None
    
    def setup_model(self):
        """Setup base model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logger.info("Using HuggingFace token for authentication")
        else:
            logger.warning("No HuggingFace token found. May not be able to access gated models.")
        
        # Setup quantization config if using 4-bit
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "token": hf_token
        }
        
        if self.use_4bit and bnb_config:
            model_kwargs.update({
                "quantization_config": bnb_config,
                "device_map": self.device_map,
                "torch_dtype": None
            })
        else:
            # CPU or non-quantized
            model_kwargs.update({
                "device_map": self.device_map if self.device_map != "cpu" else None,
                "torch_dtype": torch.float32 if self.device_map == "cpu" else torch.bfloat16,
                "low_cpu_mem_usage": True
            })
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to CPU if needed
        if self.device_map == "cpu":
            self.model = self.model.to("cpu")
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        self._setup_lora()
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
    
    def train(
        self,
        dataset: DatasetDict,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        output_dir: Optional[str] = None,
        logging_steps: int = 1,
        save_steps: int = 100,
        eval_steps: int = 5,
        sample_prompts: Optional[List[str]] = None
    ):
        """Train the model.
        
        Args:
            dataset: Training dataset with train/validation splits
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            output_dir: Directory to save model
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            sample_prompts: Optional prompts to generate samples during training
            
        Returns:
            Trained model
        """
        if not self.model:
            self.setup_model()
        
        self.output_dir = output_dir or f"./models/{self.model_name.split('/')[-1]}-cypher"
        
        # Tokenize dataset
        tokenized_dataset = self._tokenize_dataset(dataset)
        
        # Setup training arguments
        # Setup training arguments based on device
        training_kwargs = {
            "output_dir": self.output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "eval_strategy": "steps",  # Updated parameter name
            "save_strategy": "steps",
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "report_to": "none",
            "remove_unused_columns": False,
        }
        
        # Adjust settings based on device
        if self.device_map == "cpu":
            training_kwargs.update({
                "fp16": False,
                "bf16": False,  # BF16 not supported on CPU
                "gradient_checkpointing": False,  # Can cause issues on CPU
                "optim": "adamw_torch",
                "dataloader_num_workers": 0  # Avoid multiprocessing issues
            })
        else:
            training_kwargs.update({
                "fp16": False,
                "bf16": True,
                "gradient_checkpointing": True,
                "optim": "paged_adamw_8bit" if self.use_4bit else "adamw_torch"
            })
        
        training_args = TrainingArguments(**training_kwargs)
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize callbacks
        callbacks = []
        
        # Loss plotting callback
        self.loss_callback = LossPlottingCallback()
        callbacks.append(self.loss_callback)
        
        # Sample generation callback - adaptive format based on model
        use_llama_31_format = (
            self.model_name and ("Llama-3.1" in self.model_name or "Llama-3.2" in self.model_name) and "Instruct" in self.model_name
        )
        
        if use_llama_31_format:
            # Use Llama 3.1 format
            system_prompt = "You are an expert at generating Cypher queries for Neo4j graph databases. Generate only valid Cypher queries with no explanation."
            default_prompts = [
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGenerate a Cypher query to answer this question: Who works in Engineering?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGenerate a Cypher query to answer this question: What projects are active?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGenerate a Cypher query to answer this question: Find employees with salary over 100000<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ]
        else:
            # Use simple format for TinyLlama and others
            default_prompts = [
                "Generate only a Cypher query to answer this question. Return only the query, no explanation.\n\nQuestion: Who works in Engineering?",
                "Generate only a Cypher query to answer this question. Return only the query, no explanation.\n\nQuestion: What projects are active?",
                "Generate only a Cypher query to answer this question. Return only the query, no explanation.\n\nQuestion: Find employees with salary over 100000"
            ]
        prompts_to_use = sample_prompts if sample_prompts else default_prompts
        
        self.sample_callback = SampleGenerationCallback(
            tokenizer=self.tokenizer,
            sample_prompts=prompts_to_use,
            eval_steps=max(eval_steps * 2, 20)  # Generate samples less frequently
        )
        callbacks.append(self.sample_callback)
        logger.info(f"Initialized sample generation with {len(prompts_to_use)} prompts")
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Plot and save loss curves
        if self.loss_callback:
            loss_plot_path = f"{self.output_dir}/training_loss_curve.png"
            logger.info("Generating loss curve plot...")
            self.loss_callback.plot_losses(save_path=loss_plot_path)
            logger.info(f"Loss curve saved to {loss_plot_path}")
        
        # Save sample generation history
        if self.sample_callback:
            import json
            samples_path = f"{self.output_dir}/training_samples.json"
            samples_history = self.sample_callback.get_samples_history()
            with open(samples_path, 'w') as f:
                json.dump(samples_history, f, indent=2)
            logger.info(f"Training samples history saved to {samples_path}")
            logger.info(f"Generated {len(samples_history)} sample outputs during training")
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return self.peft_model
    
    def _tokenize_dataset(
        self,
        dataset: DatasetDict,
        max_length: int = 512
    ) -> DatasetDict:
        """Tokenize the dataset.
        
        Args:
            dataset: Dataset to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Tokenize the text field
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            # Set labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ):
        """Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> str:
        """Generate Cypher query from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to sample
            
        Returns:
            Generated Cypher query
        """
        if not self.model:
            raise ValueError("Model not loaded. Call setup_model() or load_checkpoint() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated.strip()