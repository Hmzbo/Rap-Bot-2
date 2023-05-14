# Rap Bot powered by GPT 3.5 API & Tacotron 2
![screenshot](https://github.com/Hmzbo/Rap-Bot-2/blob/main/Images/Screenshot.png)

This Rap Bot can be used to generate high quality rap lyrics with interesting rythme schemes, and to synthesize vocals by using Tacotron 2 models which can be trained to clone voices. You can use the FakeYou notebooks to train and test Tacotron 2 models on google colab. You can find the notebooks in the "Extra" directory.

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN
2. Python==3.7.16
3. Tacotron 2 model. The model used in the demo notebook can be downloaded via this [link](https://drive.google.com/file/d/1-E3UBK55_JZ36GqRH196WKn2mra6SSlM/view?usp=sharing). 
    - **Tacotron  model must be in /fakeyou/ to run the Gradio application**

## Setup
1. Clone this repository `git clone https://github.com/Hmzbo/Rap-Bot-2.git`.
2. Create a new virtual or conda environment with python 3.7.16
    - To create a conda env run the following command: `conda create -n rapbot python=3.7.16`
3. Run the following command to install required dependencies `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 -U`
    - If GPU was not used by `torch`, verify the compatibility of the CUDA version with your machine GPU. By default, CUDA Toolkit 11.7 is installed through the requirements file. 
4. Create "credentials.py" file in current directory, containing your OpenAI API key:
    - example: `API_KEY_OPENAI = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
    - **Note:** You can still run the Rap Bot even without the API key, however you can't then use the `Generate Lyrics` tab.
5. Copy Tacotron 2 models into `/fakeyou/` directory.
6. You can run the Rap Bot application from within the jupyter notebook `Rapbot2.ipynb` or by running the `Rapbot2.py` from a terminal with the following command:
    - `python Rapbot2.py` with the following optional arguments:
        - `--share`: To create a shareable Gradio application link.
        - `--debug`: To activate gradio debugging mode.
        - `--show-err`: To display Errors on the UI and browser console log.

## How to use?
The user interface has the basic instructions you need to know to understand the how to use the Rap Bot. Otherwise, you can follow the tutorial [video](url)(not ready yet.).

## Credits
This project harnesses the capabilities of a variety of interesting open-source projects:
 - Gradio: https://github.com/gradio-app/gradio
 - HiFi-GAN: https://github.com/justinjohn0306/hifi-gan
 - Tacotron 2: https://github.com/justinjohn0306/tacotron2
 - gdown: https://github.com/wkentaro/gdown

As well as, the OpenAI GPT 3.5 [API](https://platform.openai.com/docs/introduction).
 
