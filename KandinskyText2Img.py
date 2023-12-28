import os
import torch
from huggingface_hub import hf_hub_url, cached_download
from copy import deepcopy
from omegaconf.dictconfig import DictConfig

from kandinsky2.configs import CONFIG_2_0, CONFIG_2_1
from kandinsky2.kandinsky2_model import Kandinsky2
from kandinsky2.kandinsky2_1_model import Kandinsky2_1
from kandinsky2.kandinsky2_2_model import Kandinsky2_2
from kandinsky3 import Kandinsky3Pipeline

ckpt_dir = "weights"

def get_kandinsky2_0(device, task_type = "text2img", use_auth_token = None):
    cache_dir = os.path.join(ckpt_dir, "2_0")
    config = deepcopy(CONFIG_2_0)
    if task_type == "inpainting":
        model_name = "Kandinsky-2-0-inpainting.pt"
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.0", filename=model_name)
    elif task_type == "text2img" or task_type == "img2img":
        model_name = "Kandinsky-2-0.pt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = model_name)
    else:
        raise ValueError("Доступны только text2img, img2img и inpainting")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = model_name, use_auth_token = use_auth_token)
    cache_dir_text_en1 = os.path.join(cache_dir, "text_encoder1")
    for name in ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = f"text_encoder1/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en1, force_filename = name, use_auth_token = use_auth_token)
    cache_dir_text_en2 = os.path.join(cache_dir, "text_encoder2")
    for name in ["config.json", "pytorch_model.bin", "spiece.model", "special_tokens_map.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = f"text_encoder2/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en2, force_filename = name, use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = "vae.ckpt")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = "vae.ckpt", use_auth_token = use_auth_token)
    config["text_enc_params1"]["model_path"] = cache_dir_text_en1
    config["text_enc_params2"]["model_path"] = cache_dir_text_en2
    config["tokenizer_name1"] = cache_dir_text_en1
    config["tokenizer_name2"] = cache_dir_text_en2
    config["image_enc_params"]["params"]["ckpt_path"] = os.path.join(cache_dir, "vae.ckpt")
    unet_path = os.path.join(cache_dir, model_name)
    model = Kandinsky2(config, unet_path, device, task_type)
    return model

def get_kandinsky2_1(device, task_type = "text2img", use_auth_token=None, use_flash_attention=False):
    cache_dir = os.path.join(ckpt_dir, "2_1")
    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    if task_type == "text2img" or task_type == "img2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = model_name)
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = model_name)
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = model_name, use_auth_token = use_auth_token)
    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = prior_name)
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = prior_name, use_auth_token = use_auth_token)
    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = f"text_encoder/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en, force_filename = name, use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = "movq_final.ckpt")
    cached_download(config_file_url, cache_dir=cache_dir, force_filename = "movq_final.ckpt", use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = "ViT-L-14_stats.th")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = "ViT-L-14_stats.th", use_auth_token = use_auth_token)
    config["tokenizer_name"] = cache_dir_text_en
    config["text_enc_params"]["model_path"] = cache_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")
    cache_model_name = os.path.join(cache_dir, model_name)
    cache_prior_name = os.path.join(cache_dir, prior_name)
    model = Kandinsky2_1(config, cache_model_name, cache_prior_name, device, task_type=task_type)
    return model

def get_kandinsky2(device, task_type = "text2img", use_auth_token = None, model_version = "2.2", use_flash_attention = False):
    if model_version == "2.0":
        model = get_kandinsky2_0(device, task_type = task_type, use_auth_token = use_auth_token)
    elif model_version == "2.1":
        model = get_kandinsky2_1(device, task_type = task_type, use_auth_token = use_auth_token, use_flash_attention = use_flash_attention)
    elif model_version == "2.2":
        model = Kandinsky2_2(device = device, task_type = task_type)
    else:
        raise ValueError("Доступны только 2.0, 2.1 и 2.2")
    return model

def Kandinsky_text_to_image(prompt, opt):
    if opt["low_vram_mode"] == True:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ver_res = {
        "Kandinsky2.0": 512,
        "Kandinsky2.1": 768,
        "Kandinsky2.2": 1024,
        "Kandinsky3.0": 1024
    }
    if opt["use_custom_res"] == True:
        width = opt["w"]
        height = opt["h"]
    else:
        width = ver_res[opt["version"]]
        height = width
    if opt["version"] == "Kandinsky2.0":
        model = get_kandinsky2(device, task_type = "text2img", model_version = "2.0", use_flash_attention = opt["use_flash_attention"])
        binary_data_list = model.generate_text2img(prompt, batch_size = opt["num_samples"], h = height, w = width, num_steps = opt["steps"], denoised_type = opt["denoised_type"], dynamic_threshold_v = opt["dynamic_threshold_v"], sampler = opt["sampler"], ddim_eta = opt["eta"], guidance_scale = opt["scale"], seed = opt["seed"])
    elif opt["version"] == "Kandinsky2.1":
        model = get_kandinsky2(device = device, task_type = "text2img", model_version = "2.1", use_flash_attention = opt["use_flash_attention"])
        binary_data_list = model.generate_text2img(prompt, num_steps = opt["steps"], batch_size = opt["num_samples"], guidance_scale = opt["scale"], h = height, w = width, sampler = opt["sampler"], ddim_eta = opt["eta"], prior_cf_scale = opt["prior_scale"], prior_steps = str(opt["prior_steps"]), negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], progress = opt["progress"])
    elif opt["version"] == "Kandinsky2.2":
        if opt["ControlNET"] == True:
            model = get_kandinsky2(device, task_type = "text2imgCN", model_version = "2.2")
            binary_data_list = model.generate_text2imgCN(prompt, decoder_steps = opt["steps"], batch_size = opt["num_samples"], prior_steps = opt["prior_steps"], prior_guidance_scale = opt["prior_scale"], decoder_guidance_scale = opt["scale"], h = height, w = width, negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"])
        else:
            model = get_kandinsky2(device, task_type = "text2img", model_version = "2.2")
            binary_data_list = model.generate_text2img(prompt, decoder_steps = opt["steps"], batch_size = opt["num_samples"], prior_steps = opt["prior_steps"], prior_guidance_scale = opt["prior_scale"], decoder_guidance_scale = opt["scale"], h = height, w = width, negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"])
    elif opt["version"] == "Kandinsky3.0":
        model = Kandinsky3Pipeline.from_pretrained("weights\\3_0", variant = "fp16", torch_dtype = torch.float16, cache_dir = "weights\\3_0", device_map = None, low_cpu_mem_usage = False)
        model.enable_model_cpu_offload()
        generator = torch.Generator(device = "cpu").manual_seed(opt["seed"])
        binary_data_list = model.generate_text2img(prompt = prompt, num_inference_steps = opt["steps"], guidance_scale = opt["scale"], negative_prompt = opt["negative_prompt"], num_images_per_prompt = opt["num_samples"], height = height, width = width, generator = generator, prompt_embeds = None, negative_prompt_embeds = None, output_type = "bd", return_dict = True, callback = None, callback_steps = 1, latents = None, device = device).images
    else:
        raise ValueError("Доступны только версии Kandinsky2.0, Kandinsky2.1, Kandinsky2.2 и Kandinsky3.0")
    torch.cuda.empty_cache()
    return binary_data_list

if __name__ == "__main__":
    params = {
        "version": "Kandinsky3.0",              #("Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2", "Kandinsky3.0")
        "ControlNET": False,                    #Только для "Kandinsky2.2"
        "negative_prior_prompt": "",            #Только для 2.0 < Kandinsky < 3.0
        "negative_prompt": "",                  #Только для Kandinsky > 2.0
        "prior_steps": 25,                      #Только для 2.0 < Kandinsky < 3.0
        "steps": 50,                            #Количество шагов обработки (от 2 до 1000)
        "seed": 42,                             #Инициализирующее значение (может быть от 0 до 2147483647)
        "use_custom_res": False,                #Использовать собственное разрешение изображения для каждой модели, вместо рекомендованного
        "w": 512,                               #512 для Kandinsky 2.0, 768 для Kandinsky 2.1 и 1024 для Kandinsky 2.2-3.0 (только для "use_custom_res": True)
        "h": 512,                               #512 для Kandinsky 2.0, 768 для Kandinsky 2.1 и 1024 для Kandinsky 2.2-3.0 (только для "use_custom_res": True)
        "sampler": "ddim_sampler",              #("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky < 2.2
        "num_samples": 1,                       #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
        "eta": 0.05,                            #только для обработчика "ddim_sampler" и Kandinsky < 2.2
        "scale": 3.0,                           #От 0.1 до 30.0
        "prior_scale": 4,                       #Только для 2.0 < Kandinsky < 3.0
        "denoised_type": "dynamic_threshold",   #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "dynamic_threshold_v": 99.5,            #Только для "Kandinsky2.0" и "dynamic_threshold"
        "use_flash_attention": False,           #Только для "Kandinsky" < 3.0
        "progress": True,                       #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "low_vram_mode": False,                 #Режим для работы на малом количестве видеопамяти. Системный параметр
        "max_dim": 1048576                      #На даный момент я не могу генерировать изображения больше 1024 на 1024. Системный параметр
    }
    #prompt = "A teddy bear на красной площади"
    prompt = "A robot, 4k photo"
    params["negative_prompt"] = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
    images = Kandinsky_text_to_image(prompt, params)
    iii = 0
    for image in images:
        iii += 1
        with open("img_" + str(iii) + ".png", "wb") as f:
            f.write(image)