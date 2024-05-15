# ComfyUI_ParlerTTS
This is a simple ComfyUI custom TTS node based on Parler_tts. Now you can using TTS audio for “ComfyUI-VideoHelperSuite node”   

update:
-----
新模型parler-tts/parler-tts-mini-jenny-30H，在描述语中，需要添加关键词Jenny   
New model: parler-tts/parler-tts-mini-jenny-30H    this model must using keyword “Jenny” in the voice description:

如果你把模型保存在diffusers目录，则可以删掉repo_id的内容，然后用菜单选择模型.   
If you save the model in the diffusers directory, you can delete the content of repo_id and then select the model from the menu   

1.Notice：
------
You can find the project for Parler-TTS here: [Parler-TTS](https://github.com/huggingface/parler-tts)  It is known that there may be library file conflicts, causing some ComfyUI node to be unusabl To prevent the dependency of this plugin from affecting other comfyui plugins, I suggest the version of the "protobuf" library=3.20.3 because other versions will affect the "tensorflow-intel" library.  

已知会有库文件冲突，导致某些ComfyUI 插件无法使用，为了防止此插件的依赖影响其他comfyui插件，我建议“ protobuf”库的版本==3.20.3 因为其他版本会影响“tensorflow-intel”库   
    
2.Installation
-----
2.1 In the .\ComfyUI \ custom_node directory, run the following:  

  ``` python 
  git clone https://github.com/smthemex/ComfyUI_ParlerTTS.git
  ```
2.2 install dependencies

安装前的注意事项   
这个插件的核心库是“descript-audiotools”，其 torch需求依赖较低，如果安装此方法的安装依赖，则会导致 类似torch==2.2.2+cu121的高版本torch被卸载，所以，如果你要使用 例如torch==2.2.2+cu121的高版本torch，则需要使用pip重新安装一次torch==2.2.2+cu121   

Precautions before installation:  
The core library of this node is "descript-audiotools", and its torch dependency is relatively low. If you install the installation dependency of this method, it will cause a higher version torch like torch=2.2.2+cu121 to be uninstalled. Therefore, if you want to use a higher version torch like torch=2.2.2+cu121, you need to use pip to reinstall torch=2.2.2+cu121 once  

2.3 特殊安装情况
以下都不是最终的需求库安装解决方法 (The following are not the final solution for installing the requirement library:)      

method A:   
如果你安装的comfyUI方式是,git clone ...,也就是不是整合包,你可以使用   
If the comfyUI method you installed is git clone, That is not a portable comfyui, you can use it   

 ``` python 
 pip install git+https://github.com/huggingface/parler-tts.git
 ```
method B:  
如果你安装的comfyui是整合包,你需要在comfyui的python_embeded的目录下,执行如下代码安装依赖库 ,安装库文件,注意替换X为你的实际路径：  
if the comfyui you are installing is a portable comfyui, you need to execute the following code in the python_embed directory of comfyui to install the dependency library,please replace X with your actual path:   

``` python 
pip install -r requirements.txt --target="path like X:\ComfyUI_windows\python_embeded\Lib\site-packages"
```
requirements文字暂时未列出，我提供以下方法，安装库文件,注意替换X为你的实际路径：  
The requirements text is not currently listed. I provide the following method to install the library file, please replace X with your actual path:  

 ``` python 
 pip install descript-audiotools  --target="X:\ComfyUI_windows\python_embeded\Lib\site-packages"
 ```
 可能的步骤: pip install torch (假设你现在正在使用的是高版本的torch) ,如果protobuf库报错,安装版本==3.20.3  
 Possible steps: pip install torch (assuming you are currently using a higher version of torch), if the protobuf library reports an error, install version=3.20.3   


2.4 Download the model and use it

Methon 1    
Use the default repo_id or fill in "parler-tts/parler-tts-mini-jenny-30H"   
使用默认的repo_id 或者填写parler-tts/parler-tts-mini-jenny-30H

Methon 2  
用我另一个pipeline工具下载完整模型文件到默认的diffusers目录下，就可以使用菜单来选择模型（使用时需要删掉repo_id的内容） 
Use my other pipeline tool to download the complete model file to the default diffusers directory, and then use the menu to select the model (you need to delete the content of repo_id when using it)   

Methon 3   
在repo_id中填写你已下载模型的绝对路径，例如XX:/XXX/parler-tts/parler-tts-mini-jenny-30H ，记住要用"/"   
Fill in the absolute path of the model you have downloaded in repo_id, for example XX:/XXX/parallel tts/parallel tts mini jenny 30H, remember to use "/"  


3.Example   
-------
![](https://github.com/smthemex/ComfyUI_ParlerTTS/blob/main/example.png)  

4.Citation
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
