# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torchaudio
from huggingface_hub import snapshot_download
from .parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random
import folder_paths
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ParlerTTS_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id":("STRING", {"default": "parler-tts/parler_tts_mini_v0.1"}),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL",)
    RETURN_NAMES = ("model","tokenizer",)
    FUNCTION = "main_loader"
    CATEGORY = "Parler_TTS"

    def main_loader(self,repo_id,):
        if not repo_id:
            print("no repo,download default model'parler-tts/parler_tts_mini_v0.1' ")
            repo_id = snapshot_download("parler-tts/parler_tts_mini_v0.1")
        
        model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        return (model,tokenizer,)


class ParlerTTS_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "Hey, how are you doing today?"}),
                "description": ("STRING", {"multiline": True,
                                           "default": "A female speaker with a slightly low-pitched voice "
                                                      "delivers her words quite expressively, in a very confined "
                                                      "sounding environment with clear audio quality. "
                                                      "She speaks very fast."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "prompt2audio"
    CATEGORY = "Parler_TTS"

    def prompt2audio(self, model,tokenizer, prompt, description):
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        file_name = "Parler_TTS" + ''.join(random.choice("0123456789") for _ in range(5)) + ".wav"
        path = os.path.join(folder_paths.output_directory, file_name)
        sf.write(path, audio_arr, model.config.sampling_rate)
        print(f"saving in path {path}")
        
        waveform, sample_rate = torchaudio.load(path)
        audio= {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio,)
   


NODE_CLASS_MAPPINGS = {
    "ParlerTTS_LoadModel":ParlerTTS_LoadModel,
    "ParlerTTS_Sampler": ParlerTTS_Sampler

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParlerTTS_LoadModel":"ParlerTTS_LoadModel",
    "ParlerTTS_Sampler": "ParlerTTS_Sampler"
}
