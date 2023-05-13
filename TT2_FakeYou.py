import os
import logging
from tqdm.notebook import tqdm

from os.path import exists, join, basename, splitext
import sys
import time
import gdown
from git import Repo
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import urllib.request
from PIL import Image
import torch
import json

import resampy
import scipy.signal

import gradio as gr



def ARPA(text, thisdict, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
            else: break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError: pass
        out = (out + " " + word + end_chars).strip()
    if EOS_Token and out[-1] != ";": out += ";"
    return out

def get_hifigan(MODEL_ID, conf_name):
    from env import AttrDict
    from models import Generator
    from denoiser import Denoiser
    # Download HiFi-GAN
    hifigan_pretrained_model = './fakeyou/hifimodel_' + conf_name
    #gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)
    if MODEL_ID == 1:
        urllib.request.urlretrieve('https://github.com/Hmzbo/TTS-TT2/releases/download/Assets/Superres_Twilight_33000', hifigan_pretrained_model)
    elif MODEL_ID == "universal":
        urllib.request.urlretrieve('https://github.com/Hmzbo/TTS-TT2/releases/download/Assets/g_02500000', hifigan_pretrained_model)
    else:
        d = 'https://drive.google.com/uc?id='
        gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)
    

    # Load HiFi-GAN
    conf = os.path.join("./fakeyou/hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cuda"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser



def has_MMI(STATE_DICT):
    return any(True for x in STATE_DICT.keys() if "mi." in x)

def get_Tactron2(MODEL_ID):
    from hparams import create_hparams
    from model import Tacotron2
    # Download Tacotron2
    tacotron2_pretrained_model = f"./fakeyou/{MODEL_ID}"
    if not exists(tacotron2_pretrained_model):
        try:
            d='https://drive.google.com/uc?id='
            gdown.download(d+MODEL_ID, tacotron2_pretrained_model, quiet=False)
        except:
            raise gr.Error("Invalid Tacotron ID!")
    # Load Tacotron2 and Config
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000 # Max Duration
    hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
    model = Tacotron2(hparams)
    state_dict = torch.load(tacotron2_pretrained_model)['state_dict']
    if has_MMI(state_dict):
        raise gr.Error("This app does not currently support MMI models.")
    model.load_state_dict(state_dict)
    _ = model.cuda().eval().half()
    return model, hparams



def end_to_end_infer(text, prams_dic, progress=gr.Progress()):
    progress(0, desc="Starting...")
    try:
        model = prams_dic['model']
        hifigan = prams_dic['hifigan']
        denoiser = prams_dic['denoiser']
        h = prams_dic['h']
        h2 = prams_dic['h2']
        hifigan_sr = prams_dic['hifigan_sr']
        superres_strength = prams_dic['superres_strength']
        pronounciation_dictionary = prams_dic['pronounciation_dictionary']
        show_graphs = prams_dic['show_graphs']
    except:
        raise gr.Error("No initialized Tacotron 2 model was found!")

    from text import text_to_sequence
    from meldataset import mel_spectrogram, MAX_WAV_VALUE

    #text = ' '.join([x.strip(' ') for x in text.split("\n")])
    sr_mix_list=[]
    for i in progress.tqdm([x for x in text.split("\n") if len(x)]):
        if not pronounciation_dictionary:
            if i[-1] != ";": i=i+";" 
        else: i = ARPA(i)
        with torch.no_grad(): # save VRAM by not including gradients
            sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            if show_graphs:
                image1_array = mel_outputs_postnet.float().data.cpu().numpy()[0]
                image1_plt = plt.imshow(image1_array, aspect='auto', origin='lower', interpolation='none', cmap='inferno')
                image1_pil = Image.fromarray(np.uint8(image1_plt.get_cmap()(image1_plt.get_array())*255))
                image2_array = alignments.float().data.cpu().numpy()[0].T
                image2_plt = plt.imshow(image2_array, aspect='auto', origin='lower', interpolation='none', cmap='inferno')
                image2_pil = Image.fromarray(np.uint8(image2_plt.get_cmap()(image2_plt.get_array())*255))
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]

            # Resample to 32k
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            wave = resampy.resample(
                audio_denoised,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cuda"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y *= superres_strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize
            sr_mix = sr_mix.astype(np.int16)
            sr_mix_list.append(sr_mix)
            sample_rate = h2.sampling_rate
            #ipd.display(ipd.Audio(sr_mix.astype(np.int16), rate=h2.sampling_rate))
    return (sample_rate, np.concatenate(sr_mix_list, axis=None)), image1_pil, image2_pil




def initialize_tacotron2(params_dic, error_box, progress=gr.Progress()):

    progress((0,6), desc="Starting initialization process...")

    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('librosa').setLevel(logging.WARNING)

    try:
        if params_dic["tacotron_id"] != "":
            tacotron_id = params_dic["tacotron_id"]
    except:
        raise gr.Error("No Tacotron ID provided.")

    if params_dic["hifigan_id"] in {"", "universal"}:
        hifigan_id = "universal"
    else:
        hifigan_id = params_dic["hifigan_id"]
    
    # Check if Initialized
    if not params_dic['initialized']:
        
        git_repo_url = 'https://github.com/Hmzbo/TTS-TT2.git'
        project_name = splitext(basename(git_repo_url))[0]
        if not all(item in os.listdir('./fakeyou/') for item in ['TTS-TT2','hifi-gan']):
            # clone and install
            Repo.clone_from(git_repo_url, "./fakeyou/TTS-TT2", recursive=True)
            Repo.clone_from("https://github.com/Hmzbo/hifi-gan.git", "./fakeyou/hifigan", recursive=True)

        if not exists(project_name):
            sys.path.append('./fakeyou/hifi-gan')
            sys.path.append('./fakeyou/TTS-TT2')  
        progress((1,6), desc="TT2 and HiFi-GAN repos cloned.") # downloaded TT2 and HiFi-GAN
        
        
        from hparams import create_hparams
        from model import Tacotron2
        from layers import TacotronSTFT
        from audio_processing import griffin_lim
        from text import text_to_sequence
        from env import AttrDict
        from meldataset import mel_spectrogram, MAX_WAV_VALUE
        from models import Generator
        from denoiser import Denoiser

        d = 'https://drive.google.com/uc?id='

        progress((2,6), desc="Initialized Dependancies.") 

        # Setup Pronounciation Dictionary
        urllib.request.urlretrieve('https://github.com/Hmzbo/TTS-TT2/releases/download/Assets/merged.dict.txt',
                                    './fakeyou/merged.dict.txt')
        thisdict = {}
        for line in reversed((open('./fakeyou/merged.dict.txt', "r").read()).splitlines()):
            thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

        progress((3,6), desc="Downloaded and Set up Pronounciation Dictionary.") # Downloaded and Set up Pronounciation Dictionary
        # Download character HiFi-GAN
        hifigan, h, denoiser = get_hifigan(hifigan_id, "config_v1")
        # Download super-resolution HiFi-GAN
        hifigan_sr, h2, denoiser_sr = get_hifigan(1, "config_32k")
        progress((4,6), desc="Downloaded and Set up HiFi-GAN model.") # Downloaded and Set up HiFi-GAN

        model, hparams = get_Tactron2(tacotron_id)

        progress((5,6), desc="Downloaded and Set up Tacotron2 model.") # Downloaded and Set up Tacotron2
        # Extra Info
        model.decoder.max_decoder_steps = params_dic["max_duration"] * 80
        model.decoder.gate_threshold = params_dic["stop_threshold"]

        params_dic["initialized"] = True
        params_dic["model"] = model
        params_dic["hifigan"] = hifigan
        params_dic["denoiser"] = denoiser
        params_dic["h"] = h
        params_dic["h2"] = h2
        params_dic["hifigan_sr"] = hifigan_sr

        progress((6,6))
        return ["Initialization completed successfully!", params_dic]
    else:
        progress((6,6))
        return ["Already initialized!", params_dic, error_box]

def update_tt2_model(params_dic, tacotron_id, hifigan_id):
    if params_dic["tacotron_id"] != tacotron_id:
        model, hparams = get_Tactron2(tacotron_id)
        hifigan, h, denoiser = get_hifigan(hifigan_id, "config_v1")
        model.decoder.max_decoder_steps = params_dic["max_duration"] * 80
        model.decoder.gate_threshold = params_dic["stop_threshold"]
        params_dic["model"] = model
        params_dic["hifigan"] = hifigan
        params_dic["denoiser"] = denoiser
        params_dic["h"] = h
    return params_dic

def get_tt2_params(params_dic, error_box, tacotron_id,hifigan_id="universal",pronounciation_dict=False,
                    show_graphs=True,max_dur=20, stop_thresh=0.5, superres_strength=10):
    if tacotron_id != "":
        params_dic["tacotron_id"]=tacotron_id
    else:
        raise gr.Error("No TACOTRON2 ID provided.")
    params_dic["hifigan_id"]=hifigan_id
    params_dic["pronounciation_dictionary"]=pronounciation_dict
    params_dic["show_graphs"]=show_graphs
    params_dic["max_duration"]=max_dur
    params_dic["max_decoder_steps"]=max_dur * 80
    params_dic["stop_threshold"]=stop_thresh
    params_dic["superres_strength"]=superres_strength
    params_dic["initialized"] = False
    return params_dic, error_box

