# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from .parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random

tts_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(tts_path)
file_path = os.path.dirname(path_dir)


class PromptToAudio:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING",
                               {"default": "parler-tts/parler_tts_mini_v0.1"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hey, how are you doing today?"}),
                "description": ("STRING", {"multiline": True,
                                           "default": "A female speaker with a slightly low-pitched voice "
                                                      "delivers her words quite expressively, in a very confined "
                                                      "sounding environment with clear audio quality. "
                                                      "She speaks very fast."}),
                "get_model_online": ("BOOLEAN", {"default": True},)
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_file", "audio_info")
    FUNCTION = "prompt2audio"
    CATEGORY = "Parler_TTS"

    def prompt2audio(self, model_path, prompt, description, get_model_online):
        if not model_path:
            raise ValueError("need a model_path or repo_id")
        else:
            if not get_model_online:
                os.environ['TRANSFORMERS_OFFLINE'] = "1"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
                prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                audio_arr = generation.cpu().numpy().squeeze()

                file_name = "Audio_" + ''.join(random.choice("0123456789") for _ in range(5)) + ".wav"
                path = os.path.join(file_path, "output", file_name)
                output_path = os.path.normpath(path)
                sf.write(output_path, audio_arr, model.config.sampling_rate)

                with sf.SoundFile(output_path, mode='r') as f:
                    duration = f.frames / f.samplerate
                    audio_len = f'The duration of the audio file is: {duration} seconds.'
                    audio_file = output_path
                    audio_info = f"{output_path}\n{audio_len}\n{prompt}"
                return audio_file, audio_info
            except Exception as e:
                print(e)


NODE_CLASS_MAPPINGS = {
    "PromptToAudio": PromptToAudio

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptToAudio": "PromptToAudio"
}
