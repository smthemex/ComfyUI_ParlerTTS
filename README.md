# ComfyUI_ParlerTTS
Parler-TTS is a lightweight text-to-speech (TTS) model that can generate high-quality, natural sounding speech in the style of a given speaker (gender, pitch, speaking style, etc).This is a simple ComfyUI custom node based on Parler_tts. 

You can find the project for Parler-TTS here: [Parler-TTS](https://github.com/huggingface/parler-tts) 
--

Update:
-----
**2024/12/25**  
* fix bug for diffusers mask-attention,from issues [#11](https://github.com/smthemex/ComfyUI_ParlerTTS/issues/11)
* 修复某些版本的diffusers因为mask-attention参数缺失，可能会导致错误的问题

**Previous updates**     
*  同步A new version of ParlerTTS的最新代码，适配更高版本的transformers;
*  Synchronize the latest code of a new version of ParlerTTS and adapt to higher versions of transformers
*  two new Parler-TTS checkpoints [parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1/tree/main)   and [parler-tts-large-v1](https://huggingface.co/parler-tts/parler-tts-large-v1/tree/main)      
* fix bug and support comfyUI audio save and view...  
* 新模型parler-tts/parler-tts-mini-jenny-30H，在描述语中，需要添加关键词Jenny   
* New model: parler-tts/parler-tts-mini-jenny-30H    this model must using keyword “Jenny” in the voice description:

1.Installation
-----
 In the .\ComfyUI \ custom_node directory, run the following:  

``` 
git clone https://github.com/smthemex/ComfyUI_ParlerTTS.git
```

2 Dependencies
---
```
pip install requirements.txt

```

3 Checkpoints
---
3.1 using  huggingface_hub    
Use the default repo_id or fill in "parler-tts/parler-tts-mini-jenny-30H"  ,"parler-tts/parler-tts-mini-v1 ","parler-tts/parler-tts-large-v1"....   
使用默认的repo_id 或者填写"parler-tts/parler-tts-mini-jenny-30H",这类地址，去掉引号；   

3.2 offline  
在repo_id填写绝对地址，Fill in the absolute address in the repo id ：     
X:/XXX/XXX/parler-tts/parler-tts-mini-jenny-30H

4.Example   
-------
![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/exampleA.png)


5.Citation
------
If you found this repository useful, please consider citing this work and also the original Stability AI paper:  
```   
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

```   
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
