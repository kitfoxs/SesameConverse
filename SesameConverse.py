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
from generator import load_csm_1b, Segment

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Check for required packages and install if missing
def check_requirements():
    required_packages = [
        "speech_recognition",
        "huggingface_hub",
        "transformers",
        "accelerate",
        "bitsandbytes"  # Required for efficient loading of large models
    ]
    
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
        response = input(f"Would you like to install them? (y/n): ")
        if response.lower() in ['y', 'yes']:
            import subprocess
            for pkg in missing_packages:
                print(f"📦 Installing {pkg}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
    # Import after installation
    import speech_recognition as sr
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    return sr, hf_hub_download, AutoModelForCausalLM, AutoTokenizer

# Get required packages
sr, hf_hub_download, AutoModelForCausalLM, AutoTokenizer = check_requirements()

# Check for CUDA availability and compatibility
device = "cpu"
if torch.cuda.is_available():
    try:
        # Test a small tensor operation to verify CUDA works
        test_tensor = torch.zeros(1).cuda()
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

torch.set_grad_enabled(False)  # Ensure no gradients are computed
if device == "cpu":
    print("🖥️ Using CPU (both models will be very slow)")

# Auto-configure settings based on available resources
def configure_settings():
    settings = {
        "context_size": 8,      # Max number of conversation turns to keep
        "context_limit": 4,     # Number of turns to keep when pruning
        "max_response_length": 200,  # Max characters in responses
        "max_audio_length": 20000,   # Max milliseconds for generated audio
        "temperature": 0.7,     # Temperature for text generation
        "fallback_temp": 0.5,   # Temperature for fallback generation
        "max_history": 12,      # Max conversation history entries
        "recovery_audio_ms": 3000,  # Short audio duration for recovery
    }
    
    # Adjust settings for low-memory devices
    if device == "cpu":
        settings["context_size"] = 4
        settings["context_limit"] = 2
        settings["max_audio_length"] = 10000
        settings["recovery_audio_ms"] = 2000
    elif device == "cuda":
        # Check GPU memory and adjust accordingly
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if gpu_mem < 8:  # Less than 8GB
                settings["context_size"] = 4
                settings["context_limit"] = 2
                settings["max_audio_length"] = 10000
                print("⚠️ Very limited GPU memory detected, reducing settings significantly")
            elif gpu_mem < 16:  # Less than 16GB
                settings["context_size"] = 6
                settings["context_limit"] = 2
                settings["max_audio_length"] = 15000
                print("⚠️ Limited GPU memory detected, reducing context size")
            elif gpu_mem > 32:  # More than 32GB
                settings["context_size"] = 12
                settings["context_limit"] = 8
                settings["max_audio_length"] = 30000
        except:
            pass
    return settings

CONFIG = configure_settings()

# Progress indicator for long operations
def show_progress(stop_event, message="Loading"):
    i = 0
    symbols = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    while not stop_event.is_set():
        print(f"\r{message} {symbols[i % len(symbols)]}", end="", flush=True)
        i += 1
        time.sleep(0.1)
    print("\r" + " " * (len(message) + 2), end="", flush=True)  # Clear the line

# Download and load the CSM model with timeout
print("🔄 Loading speech model...")
progress_stop = threading.Event()
progress_thread = threading.Thread(target=show_progress, args=(progress_stop, "Loading speech model"))
progress_thread.daemon = True
progress_thread.start()

try:
    # Check for cached model first
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_pattern = os.path.join(cache_dir, "**", "ckpt.pt")
    cached_model = None
    
    import glob
    for file in glob.glob(model_pattern, recursive=True):
        if os.path.exists(file) and "sesame/csm-1b" in file:
            cached_model = file
            print(f"\rUsing cached model: {cached_model}", end="", flush=True)
            break
    
    # Set a timeout for model loading
    model_queue = queue.Queue()
    
    def load_model():
        try:
            if cached_model:
                model_path = cached_model
            else:
                model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            generator = load_csm_1b(model_path, device)
            model_queue.put((generator, generator.sample_rate))
        except Exception as e:
            model_queue.put((e, None))
    
    model_thread = threading.Thread(target=load_model)
    model_thread.daemon = True
    model_thread.start()
    
    # Wait with timeout
    model_thread.join(timeout=300)  # 5 minute timeout
    
    if model_thread.is_alive():
        progress_stop.set()
        progress_thread.join()
        print("🚨 Model loading timed out after 5 minutes")
        print("🛑 Exiting as speech model is required")
        exit(1)
    
    result = model_queue.get(block=False)
    
    if isinstance(result[0], Exception):
        progress_stop.set()
        progress_thread.join()
        print(f"🚨 Failed to load speech model: {result[0]}")
        print("🛑 Exiting as speech model is required")
        exit(1)
    
    generator, sample_rate = result
    
    progress_stop.set()
    progress_thread.join()
    print("✅ Speech model loaded successfully")
    
except Exception as e:
    progress_stop.set()
    progress_thread.join()
    print(f"🚨 Failed to load speech model: {e}")
    print("🛑 Exiting as speech model is required")
    exit(1)

# Set the Gemma 3 model
text_model_name = "google/gemma-3-12b-it"
model_type = "instruct"  # Gemma 3 uses instruction format

# Load the Gemma 3 model
print(f"🔄 Loading Gemma 3 12B...")
progress_stop = threading.Event()
progress_thread = threading.Thread(target=show_progress, args=(progress_stop, "Loading Gemma 3 12B"))
progress_thread.daemon = True
progress_thread.start()

try:
    # Import specialized modules for efficient loading
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    
    # Determine optimal quantization based on available VRAM
    use_4bit = False
    use_8bit = False
    gpu_mem = 0
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem < 24:  # Less than 24GB (full model size)
            if gpu_mem < 12:  # Very limited memory
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
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    
    # Load the model with appropriate settings
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    elif device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    
    # Load the model
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        **model_kwargs
    )
    use_external_llm = True
    
    progress_stop.set()
    progress_thread.join()
    print("✅ Gemma 3 12B loaded successfully")
except Exception as e:
    progress_stop.set()
    progress_thread.join()
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

# Function to capture live microphone input
def recognize_speech():
    microphone_list = sr.Microphone.list_microphone_names()
    if not microphone_list:
        print("🚨 No microphones detected")
        return None, None
        
    recognizer = sr.Recognizer()
    try:
        # Don't force the sample rate on the microphone - use default then resample
        with sr.Microphone() as source:
            print("\n🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                # Use timeout to prevent hanging
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("⚠️ Listening timed out - no speech detected")
                return None, None
            
        print("📝 Recognizing speech...")
        # Try Google first (primary option - better accuracy but requires internet)
        try:
            text = recognizer.recognize_google(audio)
            print(f"🗨️ You: {text}")
            return text, audio
        except sr.RequestError:
            print("⚠️ Could not reach Google Speech Recognition API")
            # Fall back to Sphinx only if Google fails
            try:
                import speech_recognition as sr_check
                # Only try Sphinx if it's actually available
                text = recognizer.recognize_sphinx(audio)
                print(f"🗨️ You (Sphinx fallback): {text}")
                return text, audio
            except (ImportError, AttributeError):
                print("⚠️ Sphinx recognition not available")
                return None, None
            except Exception as e:
                print(f"⚠️ Sphinx recognition failed: {e}")
                return None, None
        except sr.UnknownValueError:
            print("⚠️ Could not understand audio")
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
        if audio_data is None or len(audio_data.frame_data) == 0:
            print("⚠️ Empty audio data")
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
        if hasattr(audio_data, 'sample_rate') and audio_data.sample_rate and sample_rate:
            if audio_data.sample_rate != sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, 
                    orig_freq=audio_data.sample_rate, 
                    new_freq=sample_rate
                )
        # Default sample rate if not specified
        elif hasattr(audio_data, 'sample_width') and audio_data.sample_width:
            # Typical default sample rates based on width
            default_rate = 16000 if audio_data.sample_width <= 2 else 44100
            if default_rate != sample_rate:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, 
                    orig_freq=default_rate, 
                    new_freq=sample_rate
                )
        
        return audio_tensor
    except Exception as e:
        print(f"⚠️ Error converting audio to tensor: {e}")
        traceback.print_exc()
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
        if np.abs(audio_np).max() < 0.01:
            print("⚠️ Generated audio is nearly silent")
            return
        
        # Normalize if too quiet
        max_amp = np.abs(audio_np).max()
        if max_amp > 0 and max_amp < 0.1:
            audio_np = audio_np * (0.5 / max_amp)
            
        # Check if sounddevice is initialized correctly
        try:
            sd.play(audio_np, samplerate=sample_rate)
            
            # Use timeout for waiting to prevent hanging
            timeout = min(len(audio_np) / sample_rate * 1.5, 45)  # 1.5x audio length or max 45 seconds
            sd.wait(timeout=timeout)
        except Exception as sd_error:
            print(f"⚠️ Error during audio playback: {sd_error}")
            # Try to reinitialize sound device
            try:
                sd.stop()
                time.sleep(0.5)
                sd.play(audio_np, samplerate=sample_rate)
                sd.wait(timeout=timeout)
            except:
                print("⚠️ Could not recover audio playback")
    except Exception as e:
        print(f"⚠️ Error playing audio: {e}")
        # Force stop any ongoing playback
        try:
            sd.stop()
        except:
            pass

# Clean text for better speech synthesis
def clean_text_for_speech(text):
    if not text:
        return "I'm sorry, I don't have a response."
        
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
    # Gemma 3 uses a specific instruction format
    system_prompt = "You are a helpful, friendly AI assistant. Keep your responses conversational, natural and concise (1-3 sentences max). Speak as if you're having a real conversation. Avoid saying you're an AI."
    
    # Format according to Gemma 3 chat template
    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    # Use limited recent history for efficiency
    recent_history = history[-6:] if len(history) > 6 else history
    
    for message in recent_history:
        if message["role"] == "user":
            formatted_messages.append({"role": "user", "content": message["content"]})
        else:
            formatted_messages.append({"role": "model", "content": message["content"]})
    
    # Convert to the format expected by the tokenizer
    try:
        # If chat_template is available (preferred method)
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Manual formatting as fallback
            prompt = ""
            for msg in formatted_messages:
                if msg["role"] == "system":
                    prompt += f"<system>\n{msg['content']}\n</system>\n\n"
                elif msg["role"] == "user":
                    prompt += f"<user>\n{msg['content']}\n</user>\n\n"
                elif msg["role"] == "model":
                    prompt += f"<model>\n{msg['content']}\n</model>\n\n"
            
            prompt += "<model>\n"  # Add generation prompt
            
    except Exception as e:
        print(f"⚠️ Error formatting prompt: {e}")
        # Ultra simple fallback
        prompt = system_prompt + "\n\n"
        for message in recent_history:
            if message["role"] == "user":
                prompt += f"User: {message['content']}\n"
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
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(text_model.device) for k, v in inputs.items()}
            
            # Adjust generation parameters based on available memory
            max_tokens = 100
            if device == "cuda":
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_mem < 12:  # Limited memory
                    max_tokens = 50
            
            # Free memory before generation
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Generate with appropriate parameters
            with torch.no_grad():
                outputs = text_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=CONFIG["temperature"],
                    top_p=0.9,
                )
            
            # Extract generated text
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Clean up response - remove any model/assistant tags or extra content
            response = re.sub(r'</?model>|</?assistant>|^Assistant:|^Model:', '', response, flags=re.IGNORECASE)
            
            # Stop at end markers if present
            stop_markers = ["</model>", "</assistant>", "<user>", "User:"]
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
            traceback.print_exc()
            # Fall back to rule-based response
            response = get_fallback_response(user_input)
    else:
        response = get_fallback_response(user_input)
    
    # Add AI response to chat history
    chat_history.append({"role": "assistant", "content": response})
    
    # Keep chat history at a reasonable length
    if len(chat_history) > CONFIG["max_history"]:
        chat_history = chat_history[-CONFIG["max_history"]:]
        
    return response

# Function for fallback responses when model isn't available
def get_fallback_response(user_input):
    user_input = user_input.lower()
    
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
        sd.stop()
    except:
        pass
    
    # Clean up GPU memory if using CUDA
    if device == "cuda":
        try:
            if text_model is not None:
                del text_model
            del generator
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

# Real-time conversation loop
def live_conversation():
    global context
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
    if device == "cuda":
        torch.cuda.empty_cache()
    
    while conversation_active:
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
            if any(word in user_input.lower() for word in ["exit", "quit", "stop", "end", "goodbye", "bye"]):
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
                    context.append(Segment(text=user_input, speaker=USER_SPEAKER_ID, audio=user_audio))
            except Exception as e:
                print(f"⚠️ Error adding user segment to context: {e}")
                # Create a new context if needed
                if not context:
                    context = [Segment(text=user_input, speaker=USER_SPEAKER_ID, audio=None)]
            
            # Generate AI text response
            print("💭 Thinking with Gemma 3...")
            
            # Free memory before generation
            if device == "cuda":
                torch.cuda.empty_cache()
                
            ai_text_response = get_ai_response(user_input)
            print(f"💬 AI: {ai_text_response}")
            
            # Generate speech response using the CSM model
            print("🔊 Generating speech...")
            try:
                # Check if we're on GPU and running low on memory
                if device == "cuda":
                    current_mem = torch.cuda.memory_allocated()
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    if current_mem > 0.8 * total_mem:  # Using 80% threshold
                        print("⚠️ Low GPU memory, clearing cache")
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Determine what context to use
                speech_context = context
                if len(context) > 2:
                    # Use only recent context to avoid memory issues
                    speech_context = context[-2:]
                
                # Use a reasonable maximum audio length
                audio_output = generator.generate(
                    text=ai_text_response,
                    speaker=AI_SPEAKER_ID,
                    context=speech_context,
                    max_audio_length_ms=CONFIG["max_audio_length"],
                    temperature=CONFIG["temperature"],
                )
                
                # Play response
                play_audio(audio_output)
                
                # Add AI response to context
                context.append(Segment(text=ai_text_response, speaker=AI_SPEAKER_ID, audio=audio_output))
                
            except Exception as e:
                print(f"🚨 Error generating speech: {e}")
                traceback.print_exc()
                
                # Try a shorter response with minimal context
                try:
                    short_response = "I'm having trouble right now, but I'm still listening."
                    print(f"💬 AI (simplified): {short_response}")
                    
                    # Create minimal context for recovery
                    recovery_context = []
                    if context and len(context) > 0:
                        # Get just the last user message
                        for seg in reversed(context):
                            if seg.speaker == USER_SPEAKER_ID:
                                recovery_context = [Segment(text=seg.text, speaker=USER_SPEAKER_ID, audio=None)]
                                break
                    
                    if not recovery_context:
                        recovery_context = [Segment(text="Hello", speaker=USER_SPEAKER_ID, audio=None)]
                    
                    audio_output = generator.generate(
                        text=short_response,
                        speaker=AI_SPEAKER_ID,
                        context=recovery_context,
                        max_audio_length_ms=CONFIG["recovery_audio_ms"],
                        temperature=CONFIG["fallback_temp"],
                    )
                    play_audio(audio_output)
                    context.append(Segment(text=short_response, speaker=AI_SPEAKER_ID, audio=audio_output))
                except Exception as recovery_e:
                    print(f"🚨 Could not generate recovery speech: {recovery_e}")
                continue
            
            # Clean up after each turn
            if device == "cuda":
                torch.cuda.empty_cache()
                
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

if __name__ == "__main__":
    try:
        live_conversation()
    except Exception as e:
        print(f"🚨 Critical error: {e}")
        traceback.print_exc()
    finally:
        cleanup()
