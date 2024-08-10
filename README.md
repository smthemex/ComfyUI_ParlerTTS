# ComfyUI_ParlerTTS
Parler-TTS is a lightweight text-to-speech (TTS) model that can generate high-quality, natural sounding speech in the style of a given speaker (gender, pitch, speaking style, etc).
This is a simple ComfyUI custom node based on Parler_tts. 

You can find the project for Parler-TTS here: [Parler-TTS](https://github.com/huggingface/parler-tts) 
--

Update:
-----
2024/08/10    
--迭代至官方08/08的v2版代码,支持2个新模型,修复BUG,现在输出只能用comfyUI内置的audio接口;  
--add v2 code（from Parler-TTS 08/08）   
-- two new Parler-TTS checkpoints [parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1/tree/main)   and [parler-tts-large-v1](https://huggingface.co/parler-tts/parler-tts-large-v1/tree/main)      
--fix bug and support comfyUI audio save and view...  

--Previous updates   
新模型parler-tts/parler-tts-mini-jenny-30H，在描述语中，需要添加关键词Jenny   
New model: parler-tts/parler-tts-mini-jenny-30H    this model must using keyword “Jenny” in the voice description:

1.Installation
-----
 In the .\ComfyUI \ custom_node directory, run the following:  

``` 
  git clone https://github.com/smthemex/ComfyUI_ParlerTTS.git
```

2 Dependencies
---
need  "  transformers>=4.43.0  "!!!!

```
python  pip install requirements.txt

```
如果是便携包,需要在python_embeded执行以下代码:

```
python -m pip install requirements.txt
```
如果安装transformers高版本无法进入comfyUI，多半是hub报错, 在python_embeded，打开CMD中执行:   
第一步，先卸载  ：  
```
python -m pip uninstall huggingface_hub

```
第二步，在python_embeded/Lib/site-packages文件中删除huggingface开头的所有文件夹：  
第三步， 在python_embeded中，打开CMD中执行   
```
python -m pip install huggingface_hub

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
![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/example.png)  

5 My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)    
20、ComfyUI_FoleyCrafter node: [ComfyUI_FoleyCrafter](https://github.com/smthemex/ComfyUI_FoleyCrafter)


6.Citation
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
