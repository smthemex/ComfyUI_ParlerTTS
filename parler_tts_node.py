# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random

tts_path = os.path.dirname(os.path.abspath(__file__))


class ModelDownload:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING",
                            {"default": "parler-tts/parler_tts_mini_v0.1"}),
                "model_local_dir": ("STRING",
                                    {"default": "./models/diffusers"}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "display": "slider"}),
                "local_dir_use_symlinks": ("BOOLEAN", {"default": True},),
                "use_hf_mirror": ("BOOLEAN", {"default": True},)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "model_download"
    CATEGORY = "Parler_TTS"

    def hf_mirror(self, use_hf_mirror):
        if use_hf_mirror:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        else:
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
        return os.environ['HF_ENDPOINT']


    def model_download(self, repo_id, model_local_dir, max_workers, local_dir_use_symlinks, use_hf_mirror):

        self.hf_mirror(use_hf_mirror)
        from huggingface_hub import snapshot_download

        model_path = f"{model_local_dir}/{repo_id.split('/')[-1]}"  # 本地模型存储的地址

        # 开始下载
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=local_dir_use_symlinks,  # 为false时，以本地模型使用文件保存，而非blob形式保存，但是每次使用得重新下载。
            max_workers=max_workers
            # token=token,“在hugging face上生成的 自己的access token， 否则模型下载可能会中断”
            # proxies = {"https": "http://localhost:7890"},  # 可选代理端口
        )
        return (model_path,)


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

    def get_model(self, get_model_online):
        if get_model_online:
            os.environ['TRANSFORMERS_OFFLINE'] = "0"
        else:
            os.environ['TRANSFORMERS_OFFLINE'] = "1"
        return os.environ['TRANSFORMERS_OFFLINE']

    def prompt2audio(self, model_path, prompt, description, get_model_online):
        if not model_path:
            raise ValueError("need a model_path")
        else:
            self.get_model(get_model_online)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
                prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                audio_arr = generation.cpu().numpy().squeeze()

                file_name = "Audio" + ''.join(random.choice("0123456789") for _ in range(5)) + ".wav"
                path = os.path.join(tts_path, "output", file_name)
                output_path = os.path.normpath(path)
                sf.write(output_path, audio_arr, model.config.sampling_rate)

                with sf.SoundFile(output_path, mode='r') as f:
                    duration = f.frames / f.samplerate
                    audio_len = f'The duration of the audio file is: {duration} seconds.'
                    audio_file = output_path
                    audio_info = f"{output_path}\n{audio_len}\n{prompt}"
                return audio_file, audio_info
            except Exception as e:
                e = ("Notice: When using \'use_model_offline\', the model path must be the path of existing "
                     "pre downloaded models.\n 注意： 使用\'use_model_offline\'模式时，模型路径必须是已有预下载模型的路径")
                print(e)


NODE_CLASS_MAPPINGS = {
    "ModelDownload": ModelDownload,
    "PromptToAudio": PromptToAudio

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelDownload": "ModelDownload",
    "PromptToAudio": "PromptToAudio"
}
