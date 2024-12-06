# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from huggingface_hub import snapshot_download
from .parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
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

    RETURN_TYPES = ("PTTSMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Parler_TTS"

    def main_loader(self,repo_id,):
        if not repo_id:
            print("no repo,download default model'parler-tts/parler_tts_mini_v0.1' ")
            repo_id = snapshot_download("parler-tts/parler_tts_mini_v0.1")
        
        model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model={"model":model,"tokenizer":tokenizer}
        return (model,)


class ParlerTTS_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PTTSMODEL",),
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

    def prompt2audio(self, model, prompt, description):
        
        tokenizer=model.get("tokenizer")
        model = model.get("model")
        model.to(device)
        
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        waveform=torch.from_numpy(audio_arr).unsqueeze(0)
        
        audio= {"waveform": waveform.unsqueeze(0), "sample_rate": model.config.sampling_rate}
        return (audio,)
   


NODE_CLASS_MAPPINGS = {
    "ParlerTTS_LoadModel":ParlerTTS_LoadModel,
    "ParlerTTS_Sampler": ParlerTTS_Sampler

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParlerTTS_LoadModel":"ParlerTTS_LoadModel",
    "ParlerTTS_Sampler": "ParlerTTS_Sampler"
}
