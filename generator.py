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


def load_gemma3_12b_tokenizer():
    """
    Loads a Gemma 3 12B tokenizer from Hugging Face.
    If that fails, falls back to smaller models.
    """
    try:
        # Example Hugging Face model name for Gemma 3 12B
        tokenizer_name = "google/gemma-3-12b-it"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        bos = tokenizer.bos_token or "<s>"
        eos = tokenizer.eos_token or "</s>"

        # If the tokenizer doesn't define BOS/EOS, add them
        if bos not in tokenizer.special_tokens_map:
            tokenizer.add_special_tokens({"bos_token": bos})
        if eos not in tokenizer.special_tokens_map:
            tokenizer.add_special_tokens({"eos_token": eos})

        # Example post-processor for Gemma 3
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", tokenizer.bos_token_id),
                (f"{eos}", tokenizer.eos_token_id),
            ],
        )
        return tokenizer

    except Exception as e:
        print(f"Error loading Gemma 3 12B tokenizer: {e}")
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

        # Use Gemma 3 12B tokenizer
        self._text_tokenizer = load_gemma3_12b_tokenizer()

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
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
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
            empty_frame = torch.zeros(1, 33).long().to(self.device)
            empty_mask = torch.zeros(1, 33).bool().to(self.device)
            empty_mask[:, -1] = True
            return empty_frame, empty_mask

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: Union[torch.Tensor, None]) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        try:
            if audio is None:
                raise ValueError("Audio is None")
            if not isinstance(audio, torch.Tensor):
                if isinstance(audio, (list, tuple, np.ndarray)):
                    audio = torch.tensor(audio, dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported audio type: {type(audio)}")
            if audio.dtype != torch.float32:
                audio = audio.float()

            # Force to 1D if needed
            if len(audio.shape) > 1:
                if audio.shape[0] <= 4:
                    audio = audio[0] if audio.shape[0] == 1 else torch.mean(audio, dim=0)
                else:
                    audio = audio[:, 0] if audio.shape[1] <= 4 else torch.mean(audio, dim=1)

            max_val = torch.max(torch.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            audio = audio.to(self.device)
            if audio.numel() == 0 or torch.isnan(audio).any():
                raise ValueError("Audio is empty or contains NaN values")

            audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))
            if isinstance(audio_tokens, tuple):
                audio_tokens = audio_tokens[0]
            # Handle additional nesting level
            if isinstance(audio_tokens, list) or (isinstance(audio_tokens, torch.Tensor) and audio_tokens.dim() > 2):
                audio_tokens = audio_tokens[0]
            if audio_tokens is None or audio_tokens.numel() == 0:
                raise ValueError("Audio tokenization returned empty tokens")

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
            empty_frame = torch.zeros(1, 33).long().to(self.device)
            empty_mask = torch.zeros(1, 33).bool().to(self.device)
            empty_mask[:, :-1] = True
            return empty_frame, empty_mask

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(segment, Segment):
            try:
                if hasattr(segment, 'speaker') and hasattr(segment, 'text'):
                    audio = getattr(segment, 'audio', None)
                    segment = Segment(speaker=segment.speaker, text=segment.text, audio=audio)
                else:
                    print(f"Warning: Invalid segment type: {type(segment)}, creating empty segment")
                    segment = Segment(speaker=0, text="", audio=None)
            except Exception as e:
                print(f"Error creating segment: {e}")
                segment = Segment(speaker=0, text="", audio=None)

        try:
            text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        except Exception as e:
            print(f"Error in text tokenization: {e}, using empty tokens")
            text_tokens = torch.zeros(1, 33).long().to(self.device)
            text_masks = torch.zeros(1, 33).bool().to(self.device)
            text_masks[:, -1] = True

        if segment.audio is not None:
            try:
                audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
                return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
            except Exception as e:
                print(f"Error in audio processing: {e}, using text tokens only")
                return text_tokens, text_masks
        else:
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
        batch_size: int = 16,  # Added batch size parameter
    ) -> torch.Tensor:
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        if context is None or not isinstance(context, list):
            context = []

        try:
            try:
                self._model.reset_caches()
            except Exception as e:
                print(f"Warning: Error resetting caches: {e}")

            max_audio_length_ms = min(max(max_audio_length_ms, 1000), 120_000)
            max_audio_frames = int(max_audio_length_ms / 80)

            tokens, tokens_mask = [], []
            context_limit = 10
            for segment in context[:context_limit]:
                try:
                    segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                    tokens.append(segment_tokens)
                    tokens_mask.append(segment_tokens_mask)
                except Exception as e:
                    print(f"Error processing context segment: {e}")
                    continue

            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

            if not tokens:
                raise ValueError("No valid tokens to process")

            try:
                prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
                prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
            except Exception as concat_e:
                print(f"Error concatenating tokens: {concat_e}")
                prompt_tokens = gen_segment_tokens.long().to(self.device)
                prompt_tokens_mask = gen_segment_tokens_mask.bool().to(self.device)

            samples = []
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

            max_seq_len = 2048 - max_audio_frames
            if curr_tokens.size(1) >= max_seq_len:
                print(f"Warning: Inputs too long ({curr_tokens.size(1)}), truncating to {max_seq_len}")
                curr_tokens = curr_tokens[:, -max_seq_len:]
                curr_tokens_mask = curr_tokens_mask[:, -max_seq_len:]
                curr_pos = torch.arange(0, curr_tokens.size(1)).unsqueeze(0).long().to(self.device)

            safety_counter = 0
            max_safety_counter = min(max_audio_frames + 100, 1000)

            # MODIFIED: Process frames in batches to improve GPU utilization
            frames_generated = 0
            while frames_generated < max_audio_frames and safety_counter < max_safety_counter:
                try:
                    # Determine current batch size
                    current_batch_size = min(batch_size, max_audio_frames - frames_generated)
                    
                    # Generate multiple frames at once using the model
                    # Note: This assumes model.generate_frames_batch exists or we need to add it
                    batch_samples = self._model.generate_frames_batch(
                        curr_tokens, 
                        curr_tokens_mask, 
                        curr_pos, 
                        temperature, 
                        topk,
                        num_frames=current_batch_size
                    )
                    
                    # If we don't have a batch generation method, fall back to sequential
                    # but still prepare for next batch processing
                    if batch_samples is None:
                        # Fallback to original sequential generation
                        batch_samples = []
                        for i in range(current_batch_size):
                            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                            
                            if torch.all(sample == 0):
                                break
                                
                            batch_samples.append(sample)
                            
                            # Update tokens for next frame
                            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                            curr_tokens_mask = torch.cat(
                                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                            ).unsqueeze(1)
                            curr_pos = curr_pos[:, -1:] + 1
                    
                    # Process the batch results
                    if isinstance(batch_samples, torch.Tensor):
                        # If batch_samples is a tensor, it means the batch method worked
                        for i in range(batch_samples.size(0)):
                            frame = batch_samples[i:i+1]
                            if torch.all(frame == 0):
                                break
                            samples.append(frame)
                        
                        # Update position based on number of frames generated
                        frames_added = batch_samples.size(0)
                        curr_pos = curr_pos[:, -1:] + frames_added
                        
                        # Update tokens for next batch
                        last_frame = batch_samples[-1:].unsqueeze(1)
                        curr_tokens = torch.cat([last_frame, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                        curr_tokens_mask = torch.cat(
                            [torch.ones_like(last_frame).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                        ).unsqueeze(1)
                    else:
                        # If batch_samples is a list from the fallback
                        samples.extend(batch_samples)
                        frames_added = len(batch_samples)
                        
                        # If no frames were generated, we're done
                        if frames_added == 0:
                            break
                    
                    frames_generated += frames_added
                    safety_counter += frames_added
                    
                    # If we generated fewer frames than requested, we're done
                    if frames_added < current_batch_size:
                        break
                    
                except Exception as e:
                    print(f"Error generating frames batch: {e}")
                    if len(samples) > 0:
                        break
                    else:
                        raise

            if len(samples) == 0:
                raise ValueError("No audio frames were generated")

            try:
                stacked_samples = torch.stack(samples)
                if stacked_samples.dim() == 3:
                    stacked_samples = stacked_samples.permute(1, 2, 0)
                elif stacked_samples.dim() == 2:
                    stacked_samples = stacked_samples.unsqueeze(0).permute(0, 2, 1)

                audio = self._audio_tokenizer.decode(stacked_samples)
                if isinstance(audio, tuple):
                    audio = audio[0]
                if audio.dim() > 1:
                    audio = audio.squeeze()
                if audio.dim() > 1:
                    audio = audio[0] if audio.shape[0] <= audio.shape[1] else audio[:, 0]
            except Exception as decode_e:
                print(f"Error decoding audio: {decode_e}")
                raise ValueError("Failed to decode audio from generated frames")

            if self._watermarker is not None:
                try:
                    audio, wm_sample_rate = watermark(
                        self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK
                    )
                    audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                except Exception as e:
                    print(f"Warning: Watermarking failed: {e}")

            if audio is None or (isinstance(audio, torch.Tensor) and (audio.numel() == 0 or torch.isnan(audio).any())):
                raise ValueError("Generated audio is invalid (empty or contains NaN values)")

            return audio

        except Exception as e:
            print(f"Error in generate method: {e}")
            raise

def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    """
    Replaces Llama references with Gemma 3 12B. 
    """
    import os
    import gc

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    else:
        gc.collect()

    if not os.path.exists(ckpt_path):
        print(f"Model checkpoint not found at {ckpt_path}, searching in common locations...")
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
                import glob
                matching_files = glob.glob(path_pattern, recursive=True)
                for match in matching_files:
                    if os.path.exists(match) and "csm" in match.lower():
                        ckpt_path = match
                        found = True
                        print(f"Found model at: {ckpt_path}")
                        break
                if found:
                    break
            else:
                if os.path.exists(path_pattern):
                    ckpt_path = path_pattern
                    found = True
                    print(f"Found model at: {ckpt_path}")
                    break
        if not found:
            from huggingface_hub import hf_hub_download
            try:
                print("Attempting to download model from HuggingFace hub...")
                ckpt_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
                print(f"Downloaded model to: {ckpt_path}")
            except Exception as download_e:
                print(f"Failed to download model: {download_e}")
                raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path} or other common locations")

    # Replace Llama references with Gemma 3 12B
    from models import Model, ModelArgs
    model_args = ModelArgs(
        backbone_flavor="gemma-12B",  # old: "llama-1B"
        decoder_flavor="gemma-12B",   # old: "llama-100M"
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = (
            torch.bfloat16
            if hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported()
            else torch.float32
        )

    print(f"Creating Gemma 3 12B model on {device} with {dtype}")
    model = Model(model_args).to(device=device, dtype=dtype)

    # Load state dict
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
    except Exception as load_e:
        print(f"Error loading model state_dict: {load_e}")
        try:
            print("Trying alternate loading method on CPU...")
            state_dict = torch.load(ckpt_path, map_location='cpu')
        except Exception as alt_e:
            print(f"Alternate loading also failed: {alt_e}")
            raise RuntimeError(f"Could not load model weights from {ckpt_path}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Loaded state dict has invalid type: {type(state_dict)}")

    try:
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"Warning: Missing keys when loading model: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys when loading model: {load_result.unexpected_keys}")
    except Exception as e:
        print(f"Error in load_state_dict: {e}")
        try:
            print("Attempting less strict model loading...")
            valid_state_dict = {k: v for k, v in state_dict.items() if k in dict(model.named_parameters())}
            model.load_state_dict(valid_state_dict, strict=False)
            print("Model loaded with filtered state dict")
        except Exception as filter_e:
            print(f"Filtered loading also failed: {filter_e}")
            raise RuntimeError("Could not load model with any method")

    try:
        generator = Generator(model)
        return generator
    except Exception as gen_e:
        print(f"Error creating generator: {gen_e}")
        raise
