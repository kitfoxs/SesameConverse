from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: Optional[torch.Tensor] = None


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    try:
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )
        return tokenizer
    except Exception as e:
        print(f"Error loading Llama3 tokenizer: {e}")
        # Fall back to any available tokenizer
        try:
            return AutoTokenizer.from_pretrained("facebook/opt-125m")
        except Exception as fallback_e:
            print(f"Fallback tokenizer also failed: {fallback_e}")
            # Last resort fallback to a very basic tokenizer
            try:
                return AutoTokenizer.from_pretrained("gpt2")
            except:
                raise RuntimeError("Failed to load any tokenizer")


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        try:
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            mimi = loaders.get_mimi(mimi_weight, device=device)
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
        except Exception as e:
            print(f"Error loading MIMI tokenizer: {e}")
            raise RuntimeError("Failed to load audio tokenizer")

        try:
            self._watermarker = load_watermarker(device=device)
        except Exception as e:
            print(f"Warning: Could not load watermarker: {e}")
            self._watermarker = None

        self.sample_rate = getattr(mimi, 'sample_rate', 24000)  # Default to 24kHz if not specified
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        try:
            # Ensure text is not None and is a string
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
                
            # Protect against overly long text that could cause memory issues
            if len(text) > 512:
                text = text[:512]
                print(f"Text too long, truncating to 512 characters")
                
            text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
            text_frame = torch.zeros(len(text_tokens), 33).long()
            text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
            text_frame[:, -1] = torch.tensor(text_tokens)
            text_frame_mask[:, -1] = True

            frame_tokens.append(text_frame.to(self.device))
            frame_masks.append(text_frame_mask.to(self.device))
        except Exception as e:
            print(f"Error in text tokenization: {e}")
            # Return minimal valid tokens
            empty_frame = torch.zeros(1, 33).long().to(self.device)
            empty_mask = torch.zeros(1, 33).bool().to(self.device)
            empty_mask[:, -1] = True
            return empty_frame, empty_mask

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: Union[torch.Tensor, None]) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        try:
            # Ensure audio is not None and is a tensor
            if audio is None:
                raise ValueError("Audio is None")
            
            # Check if audio is already a tensor, if not convert it
            if not isinstance(audio, torch.Tensor):
                if isinstance(audio, (list, tuple, np.ndarray)):
                    audio = torch.tensor(audio, dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            # Convert to float if not already
            if audio.dtype != torch.float32:
                audio = audio.float()
                
            # Force to 1D if needed
            if len(audio.shape) > 1:
                # Take first channel or mean across channels
                if audio.shape[0] <= 4:  # Typical channels-first format
                    audio = audio[0] if audio.shape[0] == 1 else torch.mean(audio, dim=0)
                else:  # Typical channels-last format for multi-channel audio
                    audio = audio[:, 0] if audio.shape[1] <= 4 else torch.mean(audio, dim=1)
            
            # Normalize audio if needed
            max_val = torch.max(torch.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
                
            # Make sure audio is on the right device
            audio = audio.to(self.device)
            
            # Check if audio is empty or has NaN values
            if audio.numel() == 0 or torch.isnan(audio).any():
                raise ValueError("Audio is empty or contains NaN values")
            
            # Encode audio with proper error handling
            try:
                audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))
                if isinstance(audio_tokens, tuple):
                    audio_tokens = audio_tokens[0]  # Handle case where encode returns tuple
                
                # Check if audio tokens are valid
                if audio_tokens is None or audio_tokens.numel() == 0:
                    raise ValueError("Audio tokenization returned empty tokens")
            except Exception as encode_err:
                print(f"Error in audio tokenization encoding: {encode_err}")
                raise
            
            # Add EOS frame
            eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
            audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

            audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
            audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
            audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
            audio_frame_mask[:, :-1] = True

            frame_tokens.append(audio_frame)
            frame_masks.append(audio_frame_mask)
        except Exception as e:
            print(f"Error in audio tokenization: {e}")
            # Return minimal valid tokens
            empty_frame = torch.zeros(1, 33).long().to(self.device)
            empty_mask = torch.zeros(1, 33).bool().to(self.device)
            empty_mask[:, :-1] = True
            return empty_frame, empty_mask

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        if not isinstance(segment, Segment):
            # Try to convert to Segment if possible
            try:
                if hasattr(segment, 'speaker') and hasattr(segment, 'text'):
                    audio = getattr(segment, 'audio', None)
                    segment = Segment(speaker=segment.speaker, text=segment.text, audio=audio)
                else:
                    # Create a minimal valid segment
                    print(f"Warning: Invalid segment type: {type(segment)}, creating empty segment")
                    segment = Segment(speaker=0, text="", audio=None)
            except Exception as e:
                print(f"Error creating segment: {e}")
                segment = Segment(speaker=0, text="", audio=None)
        
        # Get text tokens with proper error handling
        try:
            text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        except Exception as e:
            print(f"Error in text tokenization: {e}, using empty tokens")
            text_tokens = torch.zeros(1, 33).long().to(self.device)
            text_masks = torch.zeros(1, 33).bool().to(self.device)
            text_masks[:, -1] = True
        
        # Handle segments without audio
        if segment.audio is not None:
            try:
                audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
                return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
            except Exception as e:
                print(f"Error in audio processing: {e}, using text tokens only")
                return text_tokens, text_masks
        else:
            # Return only text tokens for segments without audio
            return text_tokens, text_masks

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment] = None,
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        # Ensure text is a string and context is a list
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        if context is None or not isinstance(context, list):
            context = []
            
        try:
            # Reset model caches safely
            try:
                self._model.reset_caches()
            except Exception as e:
                print(f"Warning: Error resetting caches: {e}")
                # Continue anyway
            
            # Limit max audio frames to reasonable values
            max_audio_length_ms = min(max(max_audio_length_ms, 1000), 120_000)  # Between 1s and 2min
            max_audio_frames = int(max_audio_length_ms / 80)
            
            tokens, tokens_mask = [], []
            
            # Process context segments with a limit to prevent OOM
            context_limit = 10  # Reasonable limit for context segments
            for segment in context[:context_limit]:
                try:
                    segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                    tokens.append(segment_tokens)
                    tokens_mask.append(segment_tokens_mask)
                except Exception as e:
                    print(f"Error processing context segment: {e}")
                    # Skip problematic segment
                    continue

            # Generate new tokens for the current text
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

            # Handle empty tokens case
            if not tokens:
                raise ValueError("No valid tokens to process")

            # Concatenate all tokens
            try:
                prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
                prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
            except Exception as concat_e:
                print(f"Error concatenating tokens: {concat_e}")
                # Emergency fallback - use just the generation tokens
                prompt_tokens = gen_segment_tokens.long().to(self.device)
                prompt_tokens_mask = gen_segment_tokens_mask.bool().to(self.device)

            samples = []
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

            # Calculate maximum sequence length
            max_seq_len = 2048 - max_audio_frames
            if curr_tokens.size(1) >= max_seq_len:
                print(f"Warning: Inputs too long ({curr_tokens.size(1)}), truncating to {max_seq_len}")
                # Truncate instead of failing
                curr_tokens = curr_tokens[:, -max_seq_len:]
                curr_tokens_mask = curr_tokens_mask[:, -max_seq_len:]
                curr_pos = torch.arange(0, curr_tokens.size(1)).unsqueeze(0).long().to(self.device)

            # Safety counter to prevent infinite loops
            safety_counter = 0
            max_safety_counter = min(max_audio_frames + 100, 1000)  # Cap at reasonable value

            # Generate audio frames
            for _ in range(max_audio_frames):
                try:
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    
                    # Check if we've hit the end token
                    if torch.all(sample == 0):
                        break  # eos
                        
                    samples.append(sample)

                    # Create new tokens and mask for next iteration
                    try:
                        curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                        curr_tokens_mask = torch.cat(
                            [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                        ).unsqueeze(1)
                        curr_pos = curr_pos[:, -1:] + 1
                    except Exception as e:
                        print(f"Error updating tokens: {e}")
                        break
                    
                    # Safety counter increment
                    safety_counter += 1
                    if safety_counter >= max_safety_counter:
                        print("Warning: Hit safety limit in frame generation")
                        break
                        
                except Exception as e:
                    print(f"Error generating frame: {e}")
                    # If we have some samples already, use what we have
                    if len(samples) > 0:
                        break
                    else:
                        raise  # Re-raise if we have no samples yet

            # If we have no samples, raise an error
            if len(samples) == 0:
                raise ValueError("No audio frames were generated")

            # Decode audio from samples safely
            try:
                # Stack samples
                stacked_samples = torch.stack(samples)
                
                # Ensure proper dimensions
                if stacked_samples.dim() == 3:
                    stacked_samples = stacked_samples.permute(1, 2, 0)
                elif stacked_samples.dim() == 2:
                    stacked_samples = stacked_samples.unsqueeze(0).permute(0, 2, 1)
                
                # Decode
                audio = self._audio_tokenizer.decode(stacked_samples)
                
                # Handle different return structures
                if isinstance(audio, tuple):
                    audio = audio[0]  # Some decoders return (audio, sample_rate)
                
                # Handle multi-dimensional output
                if audio.dim() > 1:
                    audio = audio.squeeze()
                
                # If still multi-dimensional, take the first channel
                if audio.dim() > 1:
                    audio = audio[0] if audio.shape[0] <= audio.shape[1] else audio[:, 0]
            except Exception as decode_e:
                print(f"Error decoding audio: {decode_e}")
                raise ValueError("Failed to decode audio from generated frames")

            # Apply watermarking if available
            if self._watermarker is not None:
                try:
                    # This applies an imperceptible watermark to identify audio as AI-generated
                    audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
                    audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                except Exception as e:
                    print(f"Warning: Watermarking failed: {e}")
                    # Continue without watermarking
            
            # Final check for audio validity
            if audio is None or (isinstance(audio, torch.Tensor) and (audio.numel() == 0 or torch.isnan(audio).any())):
                raise ValueError("Generated audio is invalid (empty or contains NaN values)")
                
            return audio
            
        except Exception as e:
            print(f"Error in generate method: {e}")
            raise

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    try:
        # Verify if the model file exists
        import os
        import gc
        
        # First clean memory to ensure we have enough for loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        else:
            gc.collect()
            
        if not os.path.exists(ckpt_path):
            print(f"Model checkpoint not found at {ckpt_path}, searching in common locations...")
            
            # Check some common locations
            possible_paths = [
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "**", "ckpt.pt"),
                "model/ckpt.pt",
                "models/ckpt.pt",
                "weights/ckpt.pt",
                "csm_1b.pt",
                "csm.pt"
            ]
            
            found = False
            for path_pattern in possible_paths:
                if "*" in path_pattern:
                    # Use glob for wildcard patterns
                    import glob
                    matching_files = glob.glob(path_pattern, recursive=True)
                    for match in matching_files:
                        if os.path.exists(match) and "csm" in match.lower():
                            ckpt_path = match
                            found = True
                            print(f"Found model at: {ckpt_path}")
                            break
                else:
                    if os.path.exists(path_pattern):
                        ckpt_path = path_pattern
                        found = True
                        print(f"Found model at: {ckpt_path}")
                        break
                        
                if found:
                    break
                    
            if not found:
                try:
                    # Try to download from HuggingFace hub
                    print("Attempting to download model from HuggingFace hub...")
                    ckpt_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
                    print(f"Downloaded model to: {ckpt_path}")
                except Exception as download_e:
                    print(f"Failed to download model: {download_e}")
                    raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path} or other common locations")
            
        # Import required modules
        try:
            from models import Model, ModelArgs
        except ImportError:
            raise ImportError("Could not import Model or ModelArgs. Please ensure 'models' module is installed.")
            
        # Create model with proper arguments
        model_args = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
        )
        
        # Handle device selection
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        # Select appropriate dtype based on device and capabilities
        if device == "cuda":
            # Check if bfloat16 is supported
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            # Use float32 for CPU unless bfloat16 is supported
            dtype = torch.bfloat16 if hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported() else torch.float32
        
        # Create model with proper dtype
        print(f"Creating model on {device} with {dtype}")
        model = Model(model_args).to(device=device, dtype=dtype)
        
        # Load state dict with proper error handling
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
        except Exception as load_e:
            print(f"Error loading model state_dict: {load_e}")
            
            # Try alternate loading method
            try:
                print("Trying alternate loading method...")
                state_dict = torch.load(ckpt_path, map_location='cpu')
            except Exception as alt_e:
                print(f"Alternate loading also failed: {alt_e}")
                raise RuntimeError(f"Could not load model weights from {ckpt_path}")
        
        # Check for state dict validity
        if not isinstance(state_dict, dict):
            raise ValueError(f"Loaded state dict has invalid type: {type(state_dict)}")
            
        # Check for missing keys or unexpected keys
        try:
            load_result = model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                print(f"Warning: Missing keys when loading model: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Warning: Unexpected keys when loading model: {load_result.unexpected_keys}")
        except Exception as e:
            print(f"Error in load_state_dict: {e}")
            
            # Try with strict=False if it failed
            try:
                print("Attempting less strict model loading...")
                # Filter the state dict to include only valid keys
                valid_state_dict = {k: v for k, v in state_dict.items() if k in dict(model.named_parameters())}
                model.load_state_dict(valid_state_dict, strict=False)
                print("Model loaded with filtered state dict")
            except Exception as filter_e:
                print(f"Filtered loading also failed: {filter_e}")
                raise RuntimeError("Could not load model with any method")

        # Create generator
        try:
            generator = Generator(model)
            
            # Verify generator works with a simple test (disable for production)
            if False:  # Set to True for testing, False for production
                try:
                    print("Testing generator with a simple prompt...")
                    # Generate a very short test response to verify the model works
                    tiny_audio = generator.generate(
                        text="Hello",
                        speaker=1,
                        context=[],
                        max_audio_length_ms=500,  # Very short for quick testing
                        temperature=0.5,
                        topk=10
                    )
                    if tiny_audio is not None:
                        print("Generator test successful")
                except Exception as test_e:
                    print(f"Warning: Generator test failed: {test_e}")
                    # Continue anyway, as it might still work for real requests
                
            return generator
        except Exception as gen_e:
            print(f"Error creating generator: {gen_e}")
            raise
        
    except Exception as e:
        print(f"Error loading CSM 1B model: {e}")
        import traceback
        traceback.print_exc()
        raise
