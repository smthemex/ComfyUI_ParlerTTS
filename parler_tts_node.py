# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from .parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import sys
import random
import folder_paths
tts_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(tts_path)
file_path = os.path.dirname(path_dir)

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "spiece.model" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths != []:
    paths = [] + [x for x in paths if x]
else:
    paths = ["no PPTS model in diffusers directory", ]


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path

class PromptToAudio:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING",
                               {"default": "parler-tts/parler_tts_mini_v0.1"}),
                "model_path": (paths,),
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

    def prompt2audio(self, repo_id,model_path, prompt, description, get_model_online):
        if model_path == ["no PPTS model in diffusers directory", ] and repo_id == "":
            raise "you need fill repo_id or download PPTS model to diffusers dir "
        local_path = get_local_path(file_path, model_path)
        if repo_id == "":
            repo_id = local_path
        if not get_model_online:
            os.environ['TRANSFORMERS_OFFLINE'] = "1"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
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
                audio_info = f"{output_path}\n{audio_len}\n{prompt}"
            return (output_path, audio_info,)
        except Exception as e:
            print(e)


NODE_CLASS_MAPPINGS = {
    "PromptToAudio": PromptToAudio

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptToAudio": "PromptToAudio"
}
