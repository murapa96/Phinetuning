import os
import argparse
import logging
import time
from typing import Dict, List, Optional, Union, Any

import torch
import torch.cuda.amp as amp
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    TextIteratorStreamer
)
import threading
from datasets import load_from_disk
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(description='Enhanced text generation script')
    
    # Model loading options
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model (local directory or HF repo)')
    parser.add_argument('--tokenizer_path', type=str, 
                        help='Path to the tokenizer (defaults to model_path if not specified)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load model in 8-bit quantization for memory efficiency')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit quantization for memory efficiency')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for inference (e.g., "cuda:0", "cpu", "auto")')
    
    # Input/output options
    parser.add_argument('--input_text', type=str, 
                        help='Input text to generate from')
    parser.add_argument('--input_file', type=str, 
                        help='File containing input texts, one per line')
    parser.add_argument('--output_file', type=str,
                        help='File to save generated outputs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing multiple inputs')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter (0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--num_beams', type=int, default=1,
                        help='Number of beams for beam search (1 = greedy)')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling instead of greedy decoding')
    parser.add_argument('--streaming', action='store_true',
                        help='Stream output tokens as they are generated')
    
    # Mixed precision options
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision inference')
    parser.add_argument('--precision', type=str, default='float16', choices=['float16', 'bfloat16'],
                        help='Precision type for mixed precision inference')
    
    return parser.parse_args()

def get_device(device_arg: str) -> torch.device:
    """Determine the appropriate device for inference."""
    if device_arg != 'auto':
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

def load_model_and_tokenizer(args):
    """Load model and tokenizer with proper error handling."""
    try:
        start_time = time.time()
        logger.info(f"Loading model from {args.model_path}")
        
        # Set up quantization config if needed
        quantization_config = None
        if args.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
        # Determine device map
        device = get_device(args.device)
        device_map = {'': device.index} if device.type == 'cuda' else None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map=device_map if device_map else 'auto',
            trust_remote_code=True,
            torch_dtype=torch.float16 if args.precision == 'float16' else torch.bfloat16
        )
        
        # Load tokenizer
        tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # Handle tokenization edge cases
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f}s")
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {str(e)}", exc_info=True)
        raise

def prepare_inputs(args, tokenizer):
    """Prepare input texts for generation."""
    input_texts = []
    
    if args.input_text:
        input_texts.append(args.input_text)
        
    elif args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                if args.input_file.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        input_texts.extend([item.get('text', '') for item in data if 'text' in item])
                    elif isinstance(data, dict) and 'texts' in data:
                        input_texts.extend(data['texts'])
                else:
                    input_texts.extend([line.strip() for line in f if line.strip()])
                    
            logger.info(f"Loaded {len(input_texts)} input texts from {args.input_file}")
        except Exception as e:
            logger.error(f"Error loading input file: {str(e)}", exc_info=True)
            raise
    else:
        input_texts.append(input("Please enter text to generate from: "))
        
    return input_texts

def generate_text(model, tokenizer, input_texts, args, device):
    """Generate text using specified parameters and strategies."""
    results = []
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature if args.do_sample else None,
        top_k=args.top_k if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Process inputs in batches
    for i in range(0, len(input_texts), args.batch_size):
        batch_texts = input_texts[i:i+args.batch_size]
        try:
            # Tokenize inputs
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
            for key in inputs:
                inputs[key] = inputs[key].to(device)
                
            # Generate with optional streaming
            if args.streaming and args.batch_size == 1:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=10.0)
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    generation_config=generation_config
                )
                
                # Start generation in a separate thread
                thread = threading.Thread(target=lambda: model.generate(**generation_kwargs))
                thread.start()
                
                # Print tokens as they're generated
                print("\nGenerating: ", end="", flush=True)
                generated_text = batch_texts[0]
                for new_text in streamer:
                    generated_text += new_text
                    print(new_text, end="", flush=True)
                print("\n")
                
                results.append(generated_text)
            else:
                # Standard generation (non-streaming)
                with torch.no_grad():
                    if args.mixed_precision:
                        with amp.autocast(device_type=device.type, dtype=torch.float16 if args.precision == 'float16' else torch.bfloat16):
                            outputs = model.generate(**inputs, generation_config=generation_config)
                    else:
                        outputs = model.generate(**inputs, generation_config=generation_config)
                
                # Decode outputs
                for i, output in enumerate(outputs):
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    results.append(generated_text)
                    
                    # Print if not saving to file
                    if not args.output_file:
                        print(f"\nGenerated text {i+1}:\n{'-' * 40}\n{generated_text}\n{'-' * 40}\n")
                        
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory. Try reducing batch size or using quantization.")
            raise
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            raise
            
    return results

def save_results(results, args, input_texts):
    """Save generation results to a file."""
    if not args.output_file:
        return
        
    try:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, 'w', encoding='utf-8') as f:
            if args.output_file.endswith('.json'):
                json_output = [
                    {"input": input_text, "output": output} 
                    for input_text, output in zip(input_texts, results)
                ]
                json.dump(json_output, f, ensure_ascii=False, indent=2)
            else:
                for i, (input_text, output) in enumerate(zip(input_texts, results)):
                    f.write(f"Input {i+1}: {input_text}\n")
                    f.write(f"Output {i+1}: {output}\n")
                    f.write(f"{'-' * 60}\n")
                    
        logger.info(f"Results saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}", exc_info=True)

def main():
    """Main function with error handling."""
    args = parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(args)
        
        # Prepare input data
        input_texts = prepare_inputs(args, tokenizer)
        
        # Set model to evaluation mode
        model.eval()
        
        # Generate text
        start_time = time.time()
        results = generate_text(model, tokenizer, input_texts, args, device)
        generation_time = time.time() - start_time
        
        # Save results if requested
        save_results(results, args, input_texts)
        
        logger.info(f"Generated {len(results)} texts in {generation_time:.2f}s " 
                  f"({generation_time/len(results):.2f}s per text)")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())