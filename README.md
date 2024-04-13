# ComfyUI_ParlerTTS
This is a simple ComfyUI custom TTS node based on Parler_tts.

Notice：
------
You can find the project for Parler-TTS here:  
[Parler-TTS](https://github.com/huggingface/parler-tts) 

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
    
  Download all the files in the [link](https://huggingface.co/parler-tts/parler_tts_mini_v0.1/tree/main) and store them in a folder that you have customized.   
  
  `Method 2`  

  I need more time.  
  
  1.4 After downloading the model, copy its path and fill in the path in the node path column of comfyui  
  
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
