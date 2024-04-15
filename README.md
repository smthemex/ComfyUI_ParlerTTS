# ComfyUI_ParlerTTS
This is a simple ComfyUI custom TTS node based on Parler_tts.   
Now you can using TTS audio for “ComfyUI-VideoHelperSuite node”   

Notice：
------
You can find the project for Parler-TTS here: [Parler-TTS](https://github.com/huggingface/parler-tts) 

USING TIPS： 
-----
已知会有库文件冲突，导致某些ComfyUI 插件无法使用  
It is known that there may be library file conflicts, causing some ComfyUI node to be unusabl .
Example:

1.Installation
-----
  1.1 In the .\ComfyUI \ custom_node directory, run the following:  
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_ParlerTTS.git
  ```
  1.2 install dependencies
  
  ``` python 
  pip install git+https://github.com/huggingface/parler-tts.git
  ```

  1.3 Download the model using a browser  
  
  `Method 1`  
    
  Download all the files in the [link](https://huggingface.co/parler-tts/parler_tts_mini_v0.1/tree/main) and store them in a folder that you have customized. model_offline should be switch to True.
  
  `Method 2` 
  
  Fill in repo_id: parler tts/parler_tts_mini_v0.1 in the path, switch the model_offline switch to false, and you can directly download the model link using huggingface. If you are a user in Chinese Mainland, you can use the downloadmodel node to download the model  
  
  在路径填写repo_id：parler-tts/parler_tts_mini_v0.1，将model_offline开关切换至false，就能使用huggingface直接下载模型链接。如果你是中国大陆用户，可使用downloadmodel 节点下载模型。  
  
  1.4 Mode path  
  
 `Notice: Please refer to the usage method of "/" in the default node path for the path format of Windows`  

2.Example   
-------
![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/output/example.png)

Citation
------
If you found this repository useful, please consider citing this work and also the original Stability AI paper:  

``` python  
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```

``` python  
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
