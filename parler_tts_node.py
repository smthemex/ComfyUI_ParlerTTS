import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random


os.environ['TRANSFORMERS_OFFLINE'] = "1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts_path = BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Parler_TTS_PromptToAudio:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_in": ("STRING", {"default": "", "forceInput": True, }),
                "description_in": ("STRING", {"default": "", "forceInput": True, }),
                "model_path": ("STRING",
                               {"default": "F:/test/ComfyUI/models/diffusers/parler-tts/parler_tts_mini_v0.1"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hey, how are you doing today?"}),
                "description": ("STRING", {"multiline": True,
                                           "default": "A female speaker with a slightly low-pitched voice "
                                                      "delivers her words quite expressively, in a very confined "
                                                      "sounding environment with clear audio quality. "
                                                      "She speaks very fast."})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "parler_tts_normal"
    CATEGORY = "ComfyUI_ParlerTTS"

    def parler_tts_normal(self, prompt_in, description_in, model_path, prompt, description):
        prompt = "".join([prompt_in, prompt])
        description = "".join([description_in, prompt])
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        file_name = "Audio" + ''.join(random.choice("0123456789") for _ in range(5)) + ".wav"
        path = os.path.join(tts_path, "ComfyUI_ParlerTTS", "output", file_name)
        output_path = os.path.normpath(path)
        # print(output_path)
        sf.write(output_path, audio_arr, model.config.sampling_rate)
        # with sf.SoundFile(file_name, mode='r') as f:
            # duration = f.frames / f.samplerate
        # audio_len = f'The duration of the audio file is: {duration} seconds.'
        # print(file_name)
        prompt = f"Audio file path : {output_path}"
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "Parler_TTS_PromptToAudio": Parler_TTS_PromptToAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PParler_TTS_PromptToAudio": "Parler_TTS_PromptToAudio"
}


'''
class Parler_TTS_AudioToAudio:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_in": ("STRING", {"default": "", "forceInput": True, }),
                "description_in": ("STRING", {"default": "", "forceInput": True, }),
                "model_path": ("STRING",
                               {"default": "F:/test/ComfyUI/models/diffusers/parler-tts/parler_tts_mini_v0.1"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hey, how are you doing today?"}),
                "description": ("STRING", {"multiline": True,
                                           "default": "A female speaker with a slightly low-pitched "
                                                      "voice delivers her words quite expressively, "
                                                      "in a very confined sounding environment with "
                                                      "clear audio quality. "
                                                      "She speaks very fast."})
            }
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "parler_tts_audio"
    CATEGORY = "ComfyUI_ParlerTTS"
    
    
    def ffmpeg_suitability(self, path):
        try:
            version = subprocess.run([path, "-version"], check=True,
                                     capture_output=True).stdout.decode("utf-8")
        except:
            return 0
        score = 0
        # rough layout of the importance of various features
        simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3),
                            ("svtav1", 5), ("libopus", 1)]
        for criterion in simple_criterion:
            if version.find(criterion[0]) >= 0:
                score += criterion[1]
        # obtain rough compile year from copyright information
        copyright_index = version.find('2000-2')
        if copyright_index >= 0:
            copyright_year = version[copyright_index + 6:copyright_index + 9]
            if copyright_year.isnumeric():
                score += int(copyright_year)
        return score

    if "VHS_FORCE_FFMPEG_PATH" in os.environ:
        ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    else:
        ffmpeg_paths = []
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            imageio_ffmpeg_path = get_ffmpeg_exe()
            ffmpeg_paths.append(imageio_ffmpeg_path)
        except:
            if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
                print("Failed to import imageio_ffmpeg")
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            ffmpeg_path = imageio_ffmpeg_path
        else:
            system_ffmpeg = shutil.which("ffmpeg")
            if system_ffmpeg is not None:
                ffmpeg_paths.append(system_ffmpeg)
            if os.path.isfile("ffmpeg"):
                ffmpeg_paths.append(os.path.abspath("ffmpeg"))
            if os.path.isfile("ffmpeg.exe"):
                ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
            if len(ffmpeg_paths) == 0:
                print("No valid ffmpeg found.")
                ffmpeg_path = None
            elif len(ffmpeg_paths) == 1:
                # Evaluation of suitability isn't required, can take sole option
                # to reduce startup time
                ffmpeg_path = ffmpeg_paths[0]
            else:
                ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

    def parler_tts_audio(self, prompt_in, description_in, model_path, prompt, description):
        prompt = "".join([prompt_in, prompt])
        description = "".join([description_in, prompt])
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        file_name = "Audio" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        file_name = current_path + f"{file_name}.wav"
        sf.write(file_name, audio_arr, model.config.sampling_rate)
        return (file_name,)

    def get_audio(self, file_name, start_time=0, duration=0):
        args = [ffmpeg_path, "-v", "error", "-i", file_name]
        if start_time > 0:
            args += ["-ss", str(start_time)]
        if duration > 0:
            args += ["-t", str(duration)]
        try:
            res = subprocess.run(args + ["-f", "wav", "-"],
                                 stdout=subprocess.PIPE, check=True).stdout
        except Exception as e:
            raise Exception(f"Failed to extract audio from: {file_name}")
        return res

    def load_audio(self, audio_file, seek_seconds):
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        # Eagerly fetch the audio since the user must be using it if the
        # node executes, unlike Load Video
        audio = get_audio(audio_file, start_time=seek_seconds)
        return (lambda: audio,)


NODE_CLASS_MAPPINGS = {
    "Parler_TTS_PromptToAudio": Parler_TTS_PromptToAudio,
    "Parler_TTS_AudioToAudio": Parler_TTS_AudioToAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PParler_TTS_PromptToAudio": "Parler_TTS_PromptToAudio",
    "Parler_TTS_AudioToAudio": "Parler_TTS_AudioToAudio"
}

'''
