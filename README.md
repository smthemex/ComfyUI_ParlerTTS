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

  1.3 Download the model and use it
  
  Model download method 1: Use model_path(repo_id) download directly   
  ----
  
  ![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/output/example1.png)
    
  Model download method 2：如果无法直连huggingface  
  ---
  
 方法a,直接浏览器下载模型：   
    
 下载[link](https://huggingface.co/parler-tts/parler_tts_mini_v0.1/tree/main)所有文件到一个你喜欢的的路径，路径名不要有中文，路径所有“\”要换成“/”，在model_path填写你的路径，例如 X:/XX/XXX,关闭get_model_online,即可使用。  

 方法B,使用内置的模型下载节点：（比如你懒得去浏览器下载）  

 把主节点的model_path转为输入，然后链接模型下载节点，模型默认下载至comfyUI的diffuse目录下（你也可以改成任何你喜欢的），其他参数不要动，注意hf_mirror是开启的，不然下载速度会很慢或者直接下载不了。 等下载完成即可使用，下次再用也不用再使用节点，直接用默认的主节点即可。  

2.Example   
-------
![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/output/example.png)  
![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/output/example1.png)  

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
