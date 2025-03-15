import sounddevice as sd
import numpy as np
import torchaudio
import time
import torch
import gc
import threading
import queue
import re
import os
import sys
import warnings
import traceback
import platform
import signal
from generator import load_csm_1b, Segment
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define a global variable for graceful shutdown
shutdown_requested = False

# Cross-platform timeout handler
class TimeoutHandler:
    def __init__(self, seconds=60, error_message='Operation timed out'):
        self.seconds = seconds
        self.error_message = error_message
        self.platform = platform.system()
        
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
        
    def __enter__(self):
        # SIGALRM is not available on Windows
        if self.platform != 'Windows':
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        return self
        
    def __exit__(self, type, value, traceback):
        if self.platform != 'Windows':
            signal.alarm(0)

# Thread with timeout for Windows compatibility
def run_with_timeout(func, args=(), kwargs={}, timeout=60):
    """Run a function with a timeout, works on all platforms."""
    result_queue = queue.Queue()
    error_ref = []
    thread_finished = threading.Event()
    
    def wrapper():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            error_ref.append(e)
            result_queue.put(None)
        finally:
            thread_finished.set()
    
    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    
    # Wait for timeout or completion
    thread.join(timeout=timeout)
    
    # Check if thread is still running after timeout
    if not thread_finished.is_set():
        return None, TimeoutError(f"Function timed out after {timeout} seconds")
    
    # If thread finished but result not in queue yet, wait a bit more
    try:
        # Give a small extra time for result to appear in queue
        result = result_queue.get(block=True, timeout=1.0)
        if error_ref:
            return None, error_ref[0]
        return result, None
    except queue.Empty:
        return None, RuntimeError("Thread completed but no result was returned")

# Check for required packages and install if missing
def check_requirements():
    required_packages = {
        "speech_recognition": "SpeechRecognition",  # Package name may differ from import name
        "huggingface_hub": "huggingface_hub",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",  # Required for efficient loading of large models
        "scipy": "scipy",  # Required for audio processing
        "torch": "torch",
        "torchaudio": "torchaudio",
        "faster_whisper": "faster-whisper"  # Added for Faster Whisper speech recognition
    }
    
    missing_packages = []
    for import_name, pkg_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pkg_name)
    
    if missing_packages:
        print(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
        response = input(f"Would you like to install them? (y/n): ")
        if response.lower() in ['y', 'yes']:
            import subprocess
            for pkg in missing_packages:
                print(f"📦 Installing {pkg}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg])
                except subprocess.CalledProcessError:
                    print(f"⚠️ Failed to install {pkg}. You may need to install it manually.")
    
    # Import after installation
    import speech_recognition as sr
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Make sure SpeechRecognition has Faster Whisper integration
    if not hasattr(sr.Recognizer, 'recognize_faster_whisper'):
        print("⚠️ Your SpeechRecognition package does not have Faster Whisper integration.")
        print("⚠️ You may need to use a patched version or add the integration manually.")
        
        # Try to add Faster Whisper integration to SpeechRecognition
        try:
            from faster_whisper import WhisperModel
            
            def recognize_faster_whisper(self, audio_data, model="base", language="en", beam_size=5, **kwargs):
                """Performs speech recognition on ``audio_data`` using the Faster Whisper API.
                
                Returns the recognized text, or raises a ``speech_recognition.UnknownValueError`` exception
                if the speech is unintelligible.
                """
                assert isinstance(audio_data, sr.AudioData), "``audio_data`` must be an ``AudioData`` instance"
                
                # Convert audio to the format required by Faster Whisper
                wav_data = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16).flatten().astype(np.float32) / 32768.0
                
                # Load the Faster Whisper model
                whisper_model = WhisperModel(model, device="cuda" if torch.cuda.is_available() else "cpu")
                
                # Transcribe the audio
                segments, info = whisper_model.transcribe(
                    wav_data, 
                    beam_size=beam_size, 
                    language=language,
                    **kwargs
                )
                
                # Extract and combine the transcribed text
                result = " ".join([segment.text for segment in segments])
                
                if not result:
                    raise sr.UnknownValueError()
                    
                return result, info
                
            # Add the method to the Recognizer class
            sr.Recognizer.recognize_faster_whisper = recognize_faster_whisper
            print("✅ Successfully added Faster Whisper integration to SpeechRecognition")
            
        except Exception as e:
            print(f"⚠️ Failed to add Faster Whisper integration: {e}")
            print("⚠️ Falling back to Google Speech Recognition")
    
    return sr, hf_hub_download, AutoModelForCausalLM, AutoTokenizer

# Global exception handler to prevent crashes
def global_exception_handler(exctype, value, tb):
    global shutdown_requested
    print(f"🚨 Uncaught exception: {value}")
    traceback.print_exception(exctype, value, tb)
    shutdown_requested = True
    # Call the original exception handler
    sys.__excepthook__(exctype, value, tb)

# Set up global exception handler
sys.excepthook = global_exception_handler

# Get required packages
try:
    sr, hf_hub_download, AutoModelForCausalLM, AutoTokenizer = check_requirements()
except Exception as e:
    print(f"🚨 Fatal error during package check: {e}")
    sys.exit(1)

# Check for CUDA availability and compatibility
device = "cpu"
if torch.cuda.is_available():
    try:
        # Test a small tensor operation to verify CUDA works
        test_tensor = torch.zeros(1).cuda()
        test_tensor = test_tensor + 1  # Force a computation
        device = "cuda"
        del test_tensor
        
        # Print CUDA memory info
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_props.total_memory / 1e9
        print(f"🖥️ Using CUDA device: {gpu_props.name} with {total_memory_gb:.2f} GB memory")
        
        # Warning for small GPU memory
        if total_memory_gb < 24:
            print(f"⚠️ Warning: Gemma 3 12B requires ~24GB VRAM, but your GPU only has {total_memory_gb:.1f}GB")
            print("⚠️ The model will be heavily quantized or offloaded to CPU, which may affect performance")
    except Exception as e:
        print(f"⚠️ CUDA available but encountered an error: {e}")
        print("⚠️ Falling back to CPU")

# Set torch grad mode off globally for inference
torch.set_grad_enabled(False)

# Explicitly set torch backends for optimization
if device == "cuda":
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable memory efficient attention if available
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Enable flash attention if available
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
    except Exception as e:
        print(f"⚠️ Could not set all CUDA optimizations: {e}")
else:
    print("🖥️ Using CPU (both speech and text generation will be very slow)")
    # Try to set num threads for better CPU performance
    try:
        # Use a reasonable number of threads (not all CPUs to avoid system freezes)
        cpu_count = os.cpu_count() or 4
        num_threads = max(4, min(cpu_count - 1, 16))  # At least 4, at most 16, but leave one CPU free
        torch.set_num_threads(num_threads)
        print(f"🖥️ CPU threads: {torch.get_num_threads()}")
    except:
        pass

# Auto-configure settings based on available resources
def configure_settings():
    settings = {
        "context_size": 8,      # Max number of conversation turns to keep
        "context_limit": 4,     # Number of turns to keep when pruning
        "max_response_length": 200,  # Max characters in responses
        "max_audio_length": 20000,   # Max milliseconds for generated audio
        "temperature": 0.7,     # Temperature for text generation
        "fallback_temp": 0.5,   # Temperature for fallback generation
        "max_history": 10,      # Max conversation history entries
        "recovery_audio_ms": 3000,  # Short audio duration for recovery
        "gemma_max_tokens": 100,  # Default max tokens for Gemma responses
        "model_timeout": 300,   # Timeout for model loading (seconds)
        "speech_timeout": 180,  # Timeout for speech generation (seconds)
        "silence_threshold": 0.005,  # Threshold for detecting silent audio
        "batch_size": 16,       # Default batch size for audio frame generation
        "use_cuda_graphs": True # Enable CUDA graphs for better performance
    }
    
    # Platform-specific optimizations
    if platform.system() == 'Windows':
        # Windows often needs larger timeouts
        settings["model_timeout"] = 400
        settings["speech_timeout"] = 220
    
    # Adjust settings for CPU mode
    if device == "cpu":
        settings["context_size"] = 4
        settings["context_limit"] = 2
        settings["max_audio_length"] = 10000
        settings["recovery_audio_ms"] = 2000
        settings["gemma_max_tokens"] = 50
        settings["model_timeout"] = 600  # Longer timeout for CPU
        settings["batch_size"] = 1       # No batching on CPU
        settings["use_cuda_graphs"] = False
        print("🖥️ CPU mode: Setting batch size to 1")
    
    elif device == "cuda":
        # Automatic batch size configuration based on available GPU memory
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            free_mem = gpu_mem - (torch.cuda.memory_allocated() / 1e9)
            
            print(f"📊 Available GPU memory: {free_mem:.2f}GB of {gpu_mem:.2f}GB total")
            
            # Dynamically set batch size based on available memory
            if gpu_mem < 4:  # Very limited VRAM (<4GB)
                settings["batch_size"] = 2
                print("⚠️ Very limited GPU memory: Setting batch size to 2")
                # Also reduce other memory-intensive settings
                settings["context_size"] = 3
                settings["context_limit"] = 1
                settings["max_audio_length"] = 6000
            elif gpu_mem < 8:  # Limited VRAM (4-8GB)
                settings["batch_size"] = 4
                print("⚠️ Limited GPU memory: Setting batch size to 4")
                # Also reduce other settings
                settings["context_size"] = 4
                settings["context_limit"] = 2
                settings["max_audio_length"] = 8000
            elif gpu_mem < 12:  # Mid-range VRAM (8-12GB)
                settings["batch_size"] = 8
                print("📊 Mid-range GPU memory: Setting batch size to 8")
                settings["context_size"] = 6
                settings["max_audio_length"] = 12000
            elif gpu_mem < 16:  # Good VRAM (12-16GB)
                settings["batch_size"] = 16
                print("📊 Good GPU memory: Setting batch size to 16")
            elif gpu_mem < 24:  # High-end VRAM (16-24GB)
                settings["batch_size"] = 24
                print("📊 High-end GPU memory: Setting batch size to 24")
                settings["context_size"] = 10
                settings["max_audio_length"] = 25000
            else:  # Very high-end VRAM (>24GB)
                settings["batch_size"] = 32
                print("📊 Very high-end GPU memory: Setting batch size to 32")
                settings["context_size"] = 12
                settings["context_limit"] = 8
                settings["max_audio_length"] = 30000

            # Add more specific settings for known GPU models
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            # Specific optimizations for known GPU models
            if any(x in gpu_name for x in ["3090", "4090", "a100", "h100"]):
                # High-end GPUs with good tensor cores
                settings["batch_size"] = min(32, settings["batch_size"])
            elif any(x in gpu_name for x in ["1650", "1660", "1050", "1060"]):
                # Older GPUs may need more conservative settings
                settings["batch_size"] = min(4, settings["batch_size"])
                settings["use_cuda_graphs"] = False  # May not be well supported
            
            # If very high CUDA memory usage detected, reduce batch size
            if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.7:
                print("⚠️ High GPU memory usage detected, reducing batch size")
                settings["batch_size"] = max(2, settings["batch_size"] // 2)
                
        except Exception as e:
            print(f"⚠️ Could not detect GPU memory details: {e}")
            # Fall back to conservative default
            settings["batch_size"] = 8
    
    # Print batch size info for user
    if device == "cuda":
        print(f"🚀 Batch processing: Using batch size of {settings['batch_size']} for speech generation")
        
    return settings

CONFIG = configure_settings()

# Progress indicator for long operations
def show_progress(stop_event, message="Loading"):
    i = 0
    symbols = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    try:
        while not stop_event.is_set() and not shutdown_requested:
            print(f"\r{message} {symbols[i % len(symbols)]}", end="", flush=True)
            i += 1
            time.sleep(0.1)
    except:
        pass
    finally:
        # Always clean up the line
        print("\r" + " " * (len(message) + 10), end="", flush=True)

# Memory cleanup utility
def clean_memory():
    if device == "cuda":
        # More aggressive memory cleanup for CUDA
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # On some systems, we need a more aggressive approach
            if platform.system() != 'Windows':  # This can hang on Windows
                try:
                    with torch.cuda.device(device):
                        torch.cuda.ipc_collect()
                except Exception as e:
                    pass  # Silently ignore IPC collection failures
        except Exception as e:
            print(f"⚠️ Error during memory cleanup: {e}")
    else:
        # CPU cleanup
        gc.collect()

# Create working directory for model offloading
try:
    os.makedirs("tmp_offload", exist_ok=True)
except Exception as e:
    print(f"⚠️ Cannot create offload directory: {e}")
    # Continue anyway, model loading will fall back to other mechanisms

# Download and load the CSM model with timeout
print("🔄 Loading speech model...")
progress_stop = threading.Event()
progress_thread = threading.Thread(target=show_progress, args=(progress_stop, "Loading speech model"))
progress_thread.daemon = True
progress_thread.start()

try:
    # Check for cached model first
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        # Try to get the cache directory through the huggingface_hub if available
        try:
            from huggingface_hub import HfFolder
            hf_home = HfFolder.get_cache_dir()
        except (ImportError, AttributeError):
            # Fall back to default location if huggingface_hub can't be used
            hf_home = os.path.expanduser("~/.cache/huggingface")

    cache_dir = os.path.join(hf_home, "hub")
    model_pattern = os.path.join(cache_dir, "**", "ckpt.pt")
    cached_model = None

    # Safe glob search with timeout
    def find_model():
        try:
            import glob
            for file in glob.glob(model_pattern, recursive=True):
                if os.path.exists(file) and "sesame/csm-1b" in file:
                    return file
        except Exception as e:
            print(f"⚠️ Error searching for model cache: {e}")
        return None
    
    # Run with a timeout to prevent hanging on file system operations
    cached_model, error = run_with_timeout(find_model, timeout=30)
    if cached_model:
        print(f"\rUsing cached model: {cached_model}", end="", flush=True)
    
    def load_model():
        try:
            if cached_model:
                model_path = cached_model
            else:
                model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            
            # Clean memory before loading model
            clean_memory()
            
            generator = load_csm_1b(model_path, device)
            if hasattr(generator, 'sample_rate'):
                sample_rate = generator.sample_rate
            else:
                # Default sample rate if not available from generator
                sample_rate = 24000
                print("⚠️ Using default sample rate 24000 Hz")
            
            return generator, sample_rate
        except Exception as e:
            raise e
    
    # Use platform-agnostic timeout wrapper
    result, error = run_with_timeout(load_model, timeout=CONFIG["model_timeout"])
    
    if error:
        raise error
    
    if result is None:
        raise TimeoutError(f"Model loading timed out after {CONFIG['model_timeout']} seconds")
    
    generator, sample_rate = result
    
    # Verify generator is valid
    if generator is None:
        raise ValueError("Speech model was loaded but returned None")
    
    progress_stop.set()
    # Join with timeout to avoid hanging
    progress_thread.join(timeout=2)
    print("✅ Speech model loaded successfully")
    
except Exception as e:
    progress_stop.set()
    progress_thread.join(timeout=2)
    print(f"🚨 Failed to load speech model: {e}")
    traceback.print_exc()
    print("🛑 Exiting as speech model is required")
    sys.exit(1)

# Set the Gemma 3 model
text_model_name = "google/gemma-3-12b-it"

# Load the Gemma 3 model
print(f"🔄 Loading Gemma 3 12B...")
progress_stop = threading.Event()
progress_thread = threading.Thread(target=show_progress, args=(progress_stop, "Loading Gemma 3 12B"))
progress_thread.daemon = True
progress_thread.start()

try:
    # Import specialized modules for efficient loading
    from transformers import BitsAndBytesConfig
    
    # Determine optimal quantization based on available VRAM
    use_4bit = False
    use_8bit = False
    gpu_mem = 0
    
    if device == "cuda":
        # Clean memory before checking
        clean_memory()
        
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = gpu_mem - (torch.cuda.memory_allocated() / 1e9)
        
        print(f"📊 Available GPU memory: {free_mem:.2f}GB")
        
        if free_mem < 24:  # Less than 24GB (full model size)
            if free_mem < 10:  # Very limited memory
                use_4bit = True
                print("⚠️ Using 4-bit quantization for Gemma 3 (very limited VRAM)")
            else:
                use_8bit = True
                print("⚠️ Using 8-bit quantization for Gemma 3 (limited VRAM)")
    
    # Configure the quantization
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quantization_config = None
    
    # Define function to load tokenizer
    def load_tokenizer():
        try:
            # Create tokenizer with specific settings for Gemma 3
            tokenizer = AutoTokenizer.from_pretrained(
                text_model_name,
                padding_side="left",
                truncation_side="left"
            )
            return tokenizer
        except Exception as e:
            raise e
    
    # Load tokenizer with timeout
    tokenizer, tokenizer_error = run_with_timeout(load_tokenizer, timeout=60)
    
    if tokenizer_error:
        raise tokenizer_error
    
    if tokenizer is None:
        raise TimeoutError("Tokenizer loading timed out")
    
    # Load the model with appropriate settings
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "offload_folder": "tmp_offload"  # Offload to disk if needed
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    elif device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    
    # Define function to load model
    def load_text_model():
        try:
            # Clean memory before loading
            clean_memory()
            
            model = AutoModelForCausalLM.from_pretrained(
                text_model_name,
                **model_kwargs
            )
            return model
        except Exception as e:
            raise e
    
    # Load model with timeout
    text_model, model_error = run_with_timeout(load_text_model, timeout=CONFIG["model_timeout"])
    
    if model_error:
        raise model_error
        
    if text_model is None:
        raise TimeoutError(f"Gemma 3 model loading timed out after {CONFIG['model_timeout']} seconds")
    
    use_external_llm = True
    
    progress_stop.set()
    progress_thread.join(timeout=2)
    print("✅ Gemma 3 12B loaded successfully")
    
except Exception as e:
    progress_stop.set()
    progress_thread.join(timeout=2)
    print(f"⚠️ Could not load Gemma 3 12B: {e}")
    traceback.print_exc()
    print("⚠️ Falling back to simple response generation")
    use_external_llm = False
    text_model = None
    tokenizer = None

# Define speakers
USER_SPEAKER_ID = 0
AI_SPEAKER_ID = 1

# Store conversation history
context = []  # For speech synthesis
chat_history = []  # For text generation

# Safe interface to sounddevice
class AudioPlayer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.active_stream = None
        self.lock = threading.RLock()  # Changed from Lock to RLock to prevent deadlocks
    
    def play(self, audio_array, samplerate=None):
        with self.lock:
            # Stop any existing playback
            self.stop()
            
            # Use provided sample rate or default
            sr = samplerate if samplerate is not None else self.sample_rate
            
            # Start playback
            try:
                self.active_stream = sd.play(audio_array, samplerate=sr)
            except Exception as e:
                print(f"⚠️ Audio playback error: {e}")
                return False
        return True
    
    def stop(self):
        with self.lock:
            try:
                sd.stop()
                if self.active_stream is not None:
                    self.active_stream = None
            except Exception as e:
                print(f"⚠️ Error stopping audio: {e}")
    
    def wait(self, timeout=None):
        try:
            if timeout is not None:
                # Wait with timeout
                start_time = time.time()
                # Changed from sd.get_status().active to sd.get_stream().active
                while sd.get_stream().active and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                # Force stop if still playing
                if sd.get_stream().active:
                    self.stop()
            else:
                sd.wait()
        except Exception as e:
            print(f"⚠️ Error waiting for audio: {e}")
            self.stop()

# Initialize audio player
audio_player = AudioPlayer(sample_rate)

# Choose microphone
def select_microphone():
    try:
        microphone_list = sr.Microphone.list_microphone_names()
        if not microphone_list:
            print("🚨 No microphones detected")
            return None
            
        if len(microphone_list) == 1:
            print(f"🎤 Using microphone: {microphone_list[0]}")
            return 0
            
        print("📋 Available microphones:")
        for idx, mic_name in enumerate(microphone_list):
            print(f"{idx+1}. {mic_name}")
        
        # Use default microphone if there's no user input
        if sys.stdin.isatty():  # Only prompt if running in a terminal
            try:
                choice = input("Select microphone (or press Enter for default): ")
                if choice.strip():
                    idx = int(choice) - 1
                    if 0 <= idx < len(microphone_list):
                        print(f"🎤 Using microphone: {microphone_list[idx]}")
                        return idx
            except ValueError:
                print("⚠️ Invalid selection, using default microphone")
                
        # Default to first microphone
        print(f"🎤 Using default microphone: {microphone_list[0]}")
        return 0
    except Exception as e:
        print(f"⚠️ Error selecting microphone: {e}")
        print("🎤 Using default microphone")
        return None

# Select microphone for speech recognition
mic_index = select_microphone()

# Function to capture live microphone input
# Function to capture live microphone input using Faster Whisper
def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        # Use the selected microphone or default
        if mic_index is not None:
            try:
                with sr.Microphone(device_index=mic_index) as source:
                    print("\n🎤 Listening...")
                    # Shorter adjustment duration to prevent long waits
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    
                    try:
                        # Use timeout to prevent hanging
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                    except sr.WaitTimeoutError:
                        print("⚠️ Listening timed out - no speech detected")
                        return None, None
            except Exception as mic_error:
                print(f"⚠️ Error with selected microphone: {mic_error}")
                print("⚠️ Falling back to default microphone")
                try:
                    with sr.Microphone() as source:
                        print("\n🎤 Listening with default microphone...")
                        recognizer.adjust_for_ambient_noise(source, duration=0.3)
                        
                        try:
                            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                        except sr.WaitTimeoutError:
                            print("⚠️ Listening timed out - no speech detected")
                            return None, None
                except Exception as e:
                    print(f"⚠️ Error with default microphone: {e}")
                    return None, None
        else:
            try:
                with sr.Microphone() as source:
                    print("\n🎤 Listening...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    
                    try:
                        # Use timeout to prevent hanging
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                    except sr.WaitTimeoutError:
                        print("⚠️ Listening timed out - no speech detected")
                        return None, None
            except Exception as e:
                print(f"⚠️ Error with microphone: {e}")
                return None, None
            
        print("📝 Recognizing speech with Faster Whisper...")
        
        # Try to get the audio with a timeout to prevent hanging
        def recognize_with_whisper():
            # Configure Faster Whisper with appropriate parameters
            # You can adjust these parameters based on your needs
            whisper_kwargs = {
                "model": "base",  # Options: "tiny", "base", "small", "medium", "large"
                "language": "en",  # Language code or "auto" for auto-detection
                "beam_size": 5     # Higher values = better quality but slower
            }
            return recognizer.recognize_faster_whisper(audio, **whisper_kwargs)
            
        text_result, error = run_with_timeout(recognize_with_whisper, timeout=15)  # Increased timeout for whisper
        
        if error:
            print(f"⚠️ Whisper recognition error: {error}")
            # Fall back to Google if whisper fails
            try:
                def recognize_with_google():
                    return recognizer.recognize_google(audio)
                text, google_error = run_with_timeout(recognize_with_google, timeout=10)
                if google_error:
                    print(f"⚠️ Google fallback also failed: {google_error}")
                    return None, None
                print(f"🗨️ You (Google fallback): {text}")
                return text, audio
            except Exception as e:
                print(f"⚠️ Recognition failed: {e}")
                return None, None
        
        # Faster Whisper typically returns a tuple with the text and additional details
        if text_result and isinstance(text_result, tuple):
            text = text_result[0]  # Extract just the recognized text
        else:
            text = text_result
        
        if text and isinstance(text, str) and text.strip():
            print(f"🗨️ You: {text}")
            return text, audio
        else:
            print("⚠️ No speech recognized")
            return None, None
            
    except KeyboardInterrupt:
        raise KeyboardInterrupt("User interrupted speech recognition")
    except Exception as e:
        print(f"🚨 Error during speech recognition: {e}")
        traceback.print_exc()
        return None, None

# Function to convert speech_recognition audio to tensor
def audio_to_tensor(audio_data):
    try:
        if audio_data is None:
            print("⚠️ Empty audio data")
            return None
            
        if not hasattr(audio_data, 'frame_data') or len(audio_data.frame_data) == 0:
            print("⚠️ No frame data in audio")
            return None
            
        # Convert audio data to numpy array
        audio_np = np.frombuffer(audio_data.frame_data, dtype=np.int16)
        
        # Check if audio is too short
        if len(audio_np) < 1000:  # Arbitrary minimum length
            print("⚠️ Audio too short")
            return None
            
        # Convert to float32 and normalize
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_np)
        
        # Resample if needed and we know the rates
        sample_attr = getattr(audio_data, 'sample_rate', None)
        if sample_attr and sample_rate and sample_attr != sample_rate:
            try:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, 
                    orig_freq=sample_attr, 
                    new_freq=sample_rate
                )
            except Exception as e:
                print(f"⚠️ Resampling error: {e}")
                # Return original tensor if resampling fails
        # Default sample rate if not specified
        elif hasattr(audio_data, 'sample_width') and audio_data.sample_width:
            # Typical default sample rates based on width
            default_rate = 16000 if audio_data.sample_width <= 2 else 44100
            if default_rate != sample_rate:
                try:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, 
                        orig_freq=default_rate, 
                        new_freq=sample_rate
                    )
                except Exception as e:
                    print(f"⚠️ Default resampling error: {e}")
                    # Return original tensor if resampling fails
        
        return audio_tensor
    except Exception as e:
        print(f"⚠️ Error converting audio to tensor: {e}")
        return None

# Function to play generated speech
def play_audio(audio_tensor):
    try:
        if audio_tensor is None:
            print("⚠️ No audio to play")
            return
            
        if not isinstance(audio_tensor, torch.Tensor):
            print(f"⚠️ Expected tensor but got {type(audio_tensor)}")
            return
            
        if len(audio_tensor.shape) == 0 or audio_tensor.shape[0] == 0:
            print("⚠️ Empty audio tensor")
            return
            
        # Check if audio is too long
        if len(audio_tensor) > sample_rate * 30:  # Reduced to 30 seconds maximum
            print("⚠️ Audio too long, truncating...")
            audio_tensor = audio_tensor[:sample_rate * 30]
            
        audio_np = audio_tensor.cpu().numpy()
        
        # Check if audio is silent (very low amplitude)
        max_amp = np.abs(audio_np).max()
        if max_amp < CONFIG["silence_threshold"]:
            print(f"⚠️ Generated audio is too quiet (max amplitude: {max_amp})")
            return
        
        # Normalize audio properly for better playback
        if max_amp > 0:
            # Normalize to 0.9 to prevent clipping
            target_loudness = 0.9
            audio_np = audio_np * (target_loudness / max_amp)
            
        # Ensure audio is within bounds
        audio_np = np.clip(audio_np, -1.0, 1.0)
            
        # Play audio with global audio player
        if audio_player.play(audio_np, samplerate=sample_rate):
            # Calculate timeout based on audio length
            timeout = min(len(audio_np) / sample_rate * 1.5, 45)  # 1.5x audio length or max 45 seconds
            audio_player.wait(timeout=timeout)
            
    except Exception as e:
        print(f"⚠️ Error playing audio: {e}")
        # Force stop any ongoing playback
        audio_player.stop()

# Clean text for better speech synthesis
def clean_text_for_speech(text):
    if not text:
        return "I'm sorry, I don't have a response."
        
    # Replace smart quotes with straight ones
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    
    # Replace other problematic Unicode characters
    text = text.replace('\u2013', '-').replace('\u2014', '--')  # en-dash, em-dash
    text = text.replace('\u2026', '...').replace('\u00a0', ' ')  # ellipsis, non-breaking space
    
    # Remove any special characters that might cause issues
    text = re.sub(r'[^\w\s.,!?\'"-:;()]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Add periods to make speech flow better if ending without punctuation
    if not text[-1] in ['.', '!', '?']:
        text += '.'
    
    # Add proper spacing after punctuation if missing
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
    
    # Break very long sentences into shorter ones for better speech flow
    if len(text) > 100:
        sentences = re.split(r'([.!?])', text)
        reconstructed = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                reconstructed.append(sentences[i] + sentences[i+1])
        text = ' '.join(reconstructed)
        
    return text.strip()

# Format prompt specifically for Gemma 3
def format_gemma3_prompt(history):
    # Specific prompt format for Gemma 3
    system_prompt = "You are a helpful, friendly assistant giving spoken responses. Keep your responses conversational, natural and concise (1-3 sentences max). Avoid complex topics that would be difficult to understand when spoken. Respond in a way that sounds natural when read aloud."
    
    # Handle the case where tokenizer is not available
    if tokenizer is None:
        prompt = system_prompt + "\n\n"
        for message in history[-5:]:
            if message["role"] == "user":
                prompt += f"Human: {message['content']}\n"
            else:
                prompt += f"Assistant: {message['content']}\n"
        prompt += "Assistant: "
        return prompt
    
    # Convert chat history to the format expected by Gemma 3
    messages = []
    
    # Add system message
    messages.append({"role": "system", "content": system_prompt})
    
    # Use limited recent history (last 5 exchanges)
    recent_history = history[-5:] if len(history) > 5 else history
    
    for message in recent_history:
        role = "user" if message["role"] == "user" else "model"
        messages.append({"role": role, "content": message["content"]})
    
    try:
        # Check if the newer apply_chat_template method is available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                print(f"⚠️ Error applying chat template: {e}")
                # Fall through to backup method
        
        # Manual implementation for Gemma 3 (backup)
        # This matches Gemma 3's expected format as of early 2023
        formatted_messages = []
        
        # System message
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append(f"<system>\n{message['content']}\n</system>")
            elif message["role"] == "user":
                formatted_messages.append(f"<user>\n{message['content']}\n</user>")
            elif message["role"] == "model":
                formatted_messages.append(f"<model>\n{message['content']}\n</model>")
        
        # Add generation prompt
        formatted_messages.append("<model>")
        
        return "\n".join(formatted_messages)
        
    except Exception as e:
        print(f"⚠️ Error formatting Gemma 3 prompt: {e}")
        # Ultra simple fallback
        prompt = system_prompt + "\n\n"
        for message in recent_history:
            if message["role"] == "user":
                prompt += f"Human: {message['content']}\n"
            else:
                prompt += f"Assistant: {message['content']}\n"
        prompt += "Assistant: "
        return prompt

# Function to generate dynamic AI responses using Gemma 3
def get_ai_response(user_input):
    global chat_history
    
    # Validate input
    if not user_input or len(user_input.strip()) == 0:
        return "I didn't catch that. Could you please repeat?"
        
    # Limit input length
    max_input_len = CONFIG["max_response_length"]
    if len(user_input) > max_input_len:
        user_input = user_input[:max_input_len-3] + "..."
        print("⚠️ Input truncated due to length")
    
    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    if use_external_llm and text_model is not None and tokenizer is not None:
        try:
            # Format prompt for Gemma 3
            prompt = format_gemma3_prompt(chat_history)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(text_model.device) for k, v in inputs.items()}
            
            # Free memory before generation
            clean_memory()
            
            # Define generation function for timeout handling
            def generate_text():
                with torch.no_grad():
                    outputs = text_model.generate(
                        inputs["input_ids"],
                        max_new_tokens=CONFIG["gemma_max_tokens"],
                        do_sample=True,
                        temperature=CONFIG["temperature"],
                        top_p=0.9,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3
                    )
                    
                # Extract generated text
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return response
            
            # Generate with timeout
            response, error = run_with_timeout(generate_text, timeout=60)
            
            if error:
                print(f"⚠️ Text generation timed out or failed: {error}")
                response = get_fallback_response(user_input)
            elif response is None:
                print("⚠️ No response generated")
                response = get_fallback_response(user_input)
            else:
                # Clean up response - remove any model/assistant tags or extra content
                response = re.sub(r'</?model>|</?assistant>|^Assistant:|^Model:', '', response, flags=re.IGNORECASE)
                
                # Stop at end markers if present
                stop_markers = ["</model>", "</assistant>", "<user>", "User:", "Human:", "Human>", "<human>"]
                for marker in stop_markers:
                    if marker in response:
                        response = response.split(marker)[0]
                
                # Limit response length
                if len(response) > CONFIG["max_response_length"]:
                    response = response[:CONFIG["max_response_length"]-3] + "..."
                    
                # Clean text for better speech synthesis
                response = clean_text_for_speech(response)
                
        except Exception as e:
            print(f"⚠️ Error generating response with Gemma 3: {e}")
            # Fall back to rule-based response
            response = get_fallback_response(user_input)
    else:
        response = get_fallback_response(user_input)
    
    # Ensure we have a valid response
    if not response or len(response.strip()) == 0:
        response = "I'm thinking about what you said. Can you tell me more?"
    
    # Add AI response to chat history
    chat_history.append({"role": "assistant", "content": response})
    
    # Keep chat history at a reasonable length
    if len(chat_history) > CONFIG["max_history"]:
        chat_history = chat_history[-CONFIG["max_history"]:]
        
    return response

# Function for fallback responses when model isn't available
def get_fallback_response(user_input):
    user_input = user_input.lower() if user_input else ""
    
    # Remove common speech recognition errors/fillers
    user_input = re.sub(r'\bum\b|\buh\b|\blike\b|\byou know\b', '', user_input)
    user_input = user_input.strip()
    
    if "hello" in user_input or "hi" in user_input or "hey" in user_input:
        return "Hello! I'm your assistant. How can I help you today?"
    elif "how are you" in user_input:
        return "I'm doing well, thanks for asking! How about you?"
    elif "weather" in user_input:
        return "I don't have access to current weather data, but I'd be happy to chat about something else."
    elif "name" in user_input:
        return "I'm your assistant, powered by Gemma 3 and a conversational speech model."
    elif "thank" in user_input:
        return "You're welcome! I'm glad I could help."
    elif "goodbye" in user_input or "bye" in user_input or "see you" in user_input:
        return "It was nice talking with you. Feel free to chat again anytime!"
    elif "?" in user_input:
        return "That's an interesting question. While I don't have all the answers, I'm here to help the best I can."
    elif len(user_input) < 10:
        return "I'd love to hear more about that. Could you elaborate?"
    else:
        # Generate a contextual response
        words = user_input.split()
        if len(words) >= 3:
            first_words = " ".join(words[:3])
        else:
            first_words = user_input
        return f"I understand what you're saying about {first_words}. What else would you like to talk about?"

# Clean up function for graceful shutdown
def cleanup():
    print("🧹 Cleaning up resources...")
    
    # Stop any ongoing audio playback
    try:
        audio_player.stop()
    except:
        pass
    
    # Clean up GPU memory
    clean_memory()
    
    # Remove offload folder if it exists
    if os.path.exists("tmp_offload"):
        try:
            import shutil
            shutil.rmtree("tmp_offload")
        except Exception as e:
            print(f"⚠️ Error removing offload folder: {e}")
    
    # Clean variables
    global text_model, generator
    try:
        if 'text_model' in globals() and text_model is not None:
            del text_model
        if 'generator' in globals() and generator is not None:
            del generator
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

# Function to generate speech safely
def generate_speech_safely(text, context, max_length_ms, temp):
    if generator is None:
        print("⚠️ Speech generator is not available")
        return None
        
    try:
        # Clean memory before generation to improve performance
        clean_memory()
        
        # Validate context to prevent errors
        safe_context = []
        if context and isinstance(context, list):
            for item in context:
                if isinstance(item, Segment):
                    safe_context.append(item)
        
        # Use the run_with_timeout function for safer speech generation
        def generate_func():
            # Pass batch_size from CONFIG to enable batch processing
            return generator.generate(
                text=text,
                speaker=AI_SPEAKER_ID,
                context=safe_context,  # Ensure context is valid
                max_audio_length_ms=max_length_ms,
                temperature=temp,
                batch_size=CONFIG.get("batch_size", 16),  # Get batch size from CONFIG with fallback
            )
        
        # Increased timeout for speech generation
        output, error = run_with_timeout(
            generate_func, 
            timeout=CONFIG["speech_timeout"]
        )
        
        if error:
            print(f"⚠️ Speech generation error: {error}")
            return None
            
        # Verify output is valid
        if output is None:
            print("⚠️ Speech generation returned None")
            return None
            
        # Check if the output has any NaN values
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            print("⚠️ Speech output contains NaN values")
            return None
        
        # Ensure output is a properly shaped tensor
        if isinstance(output, torch.Tensor):
            # Remove any trailing silence
            if output.numel() > sample_rate:  # At least 1 second
                silence_threshold = 0.01
                # Find the last non-silent sample
                abs_audio = torch.abs(output)
                # Get chunks and check if they're silent
                chunk_size = sample_rate // 10  # 100ms chunks
                non_silent_chunks = []
                
                for i in range(0, len(output), chunk_size):
                    chunk = abs_audio[i:i+chunk_size]
                    if chunk.max() > silence_threshold:
                        non_silent_chunks.append(i // chunk_size)
                
                if non_silent_chunks:
                    # Keep audio up to the last non-silent chunk plus 0.5 second padding
                    last_chunk = non_silent_chunks[-1]
                    end_sample = min(len(output), (last_chunk + 1) * chunk_size + sample_rate // 2)
                    output = output[:end_sample]
            
        return output
    except Exception as e:
        print(f"⚠️ Error in generate_speech_safely: {e}")
        return None

# Real-time conversation loop
def live_conversation():
    global context, shutdown_requested
    print("🎙️ Live Voice AI is running...")
    print("Say 'exit' or 'quit' to end the conversation.")
    time.sleep(1)
    
    consecutive_errors = 0
    conversation_active = True
    
    # Print device information
    if device == "cuda":
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💡 Using GPU: {torch.cuda.get_device_name(0)} ({memory_gb:.1f} GB)")
    else:
        print("💡 Using CPU (both speech and text generation will be very slow)")
    
    print(f"💡 Speech model: CSM-1B (sample rate: {sample_rate} Hz)")
    print(f"💡 Text model: Gemma 3 12B Instruction Tuned")
    print(f"💡 Context size: {CONFIG['context_size']} turns")
    print(f"💡 Tip: Say 'exit', 'quit', or 'goodbye' to end the conversation")
    
    # Do a small cleanup before we start
    clean_memory()
    
    # Initialize context if empty
    if context is None:
        context = []
    
    while conversation_active and not shutdown_requested:
        try:
            # Capture voice input
            user_input, audio_data = recognize_speech()
            if not user_input:
                consecutive_errors += 1
                if consecutive_errors > 3:
                    print("⚠️ Multiple recognition failures. Please check your microphone.")
                    consecutive_errors = 0
                continue  # Skip empty input
            
            consecutive_errors = 0  # Reset error counter on success
            
            # Stop if user says "exit"
            exit_words = ["exit", "quit", "stop", "end", "goodbye", "bye"]
            if any(word in user_input.lower() for word in exit_words):
                print("👋 Exiting chat...")
                break
            
            # Convert speech_recognition audio to tensor if available
            user_audio = None
            if audio_data:
                user_audio = audio_to_tensor(audio_data)
            
            # Manage context size - only keep reasonable number of turns
            if len(context) > CONFIG["context_size"]:
                context = context[-CONFIG["context_limit"]:]  # Keep only last N items
                
            # Add user input to context
            try:
                if user_input and user_input.strip():  # Verify we have real text
                    new_segment = Segment(text=user_input, speaker=USER_SPEAKER_ID, audio=user_audio)
                    context.append(new_segment)
            except Exception as e:
                print(f"⚠️ Error adding user segment to context: {e}")
                # Create a new context if needed
                try:
                    if not context:
                        context = [Segment(text=user_input, speaker=USER_SPEAKER_ID, audio=None)]
                except Exception as ctx_e:
                    print(f"⚠️ Cannot create context: {ctx_e}")
                    # Last resort: empty context
                    context = []
            
            # Generate AI text response
            print("💭 Thinking with Gemma 3...")
            
            # Free memory before generation
            clean_memory()
                
            ai_text_response = get_ai_response(user_input)
            print(f"💬 AI: {ai_text_response}")
            
            # Generate speech response using the CSM model
            print("🔊 Generating speech...")
            try:
                # Check if we're on GPU and running low on memory
                if device == "cuda":
                    current_mem = torch.cuda.memory_allocated()
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    if current_mem > 0.7 * total_mem:  # Using 70% threshold (reduced from 80%)
                        print("⚠️ Low GPU memory, clearing cache")
                        clean_memory()
                
                # Safety check for context
                if context is None or not isinstance(context, list):
                    print("⚠️ Context is invalid, resetting")
                    context = []
                
                # Use safe context
                speech_context = []
                try:
                    # Use only recent context to avoid memory issues
                    if len(context) > 0:
                        if len(context) > 2:
                            speech_context = context[-2:]
                        else:
                            speech_context = context.copy()
                except Exception as e:
                    print(f"⚠️ Error copying context: {e}")
                
                # Generate speech safely with timeout handling
                audio_output = generate_speech_safely(
                    text=ai_text_response,
                    context=speech_context,
                    max_length_ms=CONFIG["max_audio_length"],
                    temp=CONFIG["temperature"]
                )
                
                # Play response if we have valid audio
                if audio_output is not None and isinstance(audio_output, torch.Tensor) and audio_output.numel() > 0:
                    play_audio(audio_output)
                    # Add AI response to context
                    try:
                        context.append(Segment(text=ai_text_response, speaker=AI_SPEAKER_ID, audio=audio_output))
                    except Exception as e:
                        print(f"⚠️ Error adding AI response to context: {e}")
                else:
                    # Speech generation failed, enter recovery
                    raise ValueError("Failed to generate valid speech output")
                
            except Exception as e:
                print(f"🚨 Error generating speech: {e}")
                
                # Try a shorter response with minimal context
                try:
                    short_response = "I'm having trouble right now, but I'm still listening."
                    print(f"💬 AI (simplified): {short_response}")
                    
                    # Create minimal context for recovery
                    recovery_context = []
                    
                    # Clean memory again
                    clean_memory()
                    
                    # Try with minimal settings and empty context
                    audio_output = generate_speech_safely(
                        text=short_response,
                        context=[],  # Use empty context for maximum reliability
                        max_length_ms=CONFIG["recovery_audio_ms"],
                        temp=CONFIG["fallback_temp"]
                    )
                    
                    # Ensure audio is valid
                    if audio_output is not None and isinstance(audio_output, torch.Tensor) and audio_output.numel() > 0:
                        play_audio(audio_output)
                        try:
                            context.append(Segment(text=short_response, speaker=AI_SPEAKER_ID, audio=audio_output))
                        except Exception as ctx_e:
                            print(f"⚠️ Error adding recovery response to context: {ctx_e}")
                    else:
                        print("⚠️ Generated recovery audio is invalid")
                        
                except Exception as recovery_e:
                    print(f"🚨 Could not generate recovery speech: {recovery_e}")
                    
                # Always add the response to context for conversation flow, even if audio failed
                if ai_text_response and ai_text_response.strip():
                    # Add to context without audio
                    try:
                        context.append(Segment(text=ai_text_response, speaker=AI_SPEAKER_ID, audio=None))
                        print("📝 Added response to context without audio")
                    except Exception as ctx_e:
                        print(f"⚠️ Error adding to context: {ctx_e}")
                        
                continue
            
            # Clean up after each turn
            clean_memory()
                
        except KeyboardInterrupt:
            print("\n👋 Exiting chat...")
            conversation_active = False
        except Exception as e:
            print(f"🚨 Error in conversation loop: {e}")
            traceback.print_exc()
            consecutive_errors += 1
            if consecutive_errors > 5:
                print("🚨 Too many consecutive errors. Exiting...")
                conversation_active = False
            # Brief pause to prevent rapid-fire errors
            time.sleep(1)

# Handle shutdown signals
def signal_handler(sig, frame):
    global shutdown_requested
    print("\n👋 Shutdown signal received, exiting...")
    shutdown_requested = True

# Register signal handlers if on a platform that supports them
if platform.system() != 'Windows':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        live_conversation()
    except Exception as e:
        print(f"🚨 Critical error: {e}")
        traceback.print_exc()
    finally:
        cleanup()
