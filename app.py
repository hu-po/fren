import logging
import os
import sys
import uuid
import numpy as np
from io import BytesIO
import gc
import torch
from typing import Dict, List, Union

import gradio as gr
import torch.nn.functional as F
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fren")
formatter = logging.Formatter("🤖|%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
fh = logging.FileHandler("fren.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
log.info(f"ROOT_DIR: {ROOT_DIR}")
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
log.info(f"KEYS_DIR: {KEYS_DIR}")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
log.info(f"DATA_DIR: {DATA_DIR}")


def set_openai_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "openai.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("OpenAI API key not found.")
            return
    os.environ["OPENAI_API_KEY"] = key
    import openai
    openai.api_key = key
    log.info("OpenAI API key set.")

def set_huggingface_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "huggingface.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("HuggingFace API key not found.")
            return
    os.environ["HUGGINGFACE_API_KEY"] = key
    log.info("HuggingFace API key set.")

def set_palm_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "palm.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Palm API key not found.")
            return
    os.environ["PALM_API_KEY"] = key
    import google.generativeai as genai
    genai.configure(api_key=key)
    log.info("Palm API key set.")

def palm_text(prompt):
    """https://developers.generativeai.google/tutorials/text_quickstart"""
    import google.generativeai as palm
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name
    print(model)

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        # The maximum length of the response
        max_output_tokens=800,
    )

    return completion.result

def palm_chat(prompt, context, examples=None):
    """https://developers.generativeai.google/tutorials/chat_quickstart"""
    import google.generativeai as palm

    # # An array of "ideal" interactions between the user and the model
    # examples = [
    #     ("What's up?", # A hypothetical user input
    #     "What isn't up?? The sun rose another day, the world is bright, anything is possible! ☀️" # A hypothetical model response
    #     ),
    #     ("I'm kind of bored",z
    #     "How can you be bored when there are so many fun, exciting, beautiful experiences to be had in the world? 🌈")
    # ]

    response = palm.chat(
    context=context,
    examples=examples,
    messages=prompt,
    )

    return response.last

def clear_gpu():
    log.info("Clearing GPU memory")
    torch.cuda.empty_cache()
    gc.collect()

def load_imagebind():
    pass

def imagebind(text, audio, image):
    sys.path.append('/home/oop/dev/ImageBind')
    import data
    from models import imagebind_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    text_list=[text]
    inputs = {
        "text": data.load_and_transform_text(text_list, device),
        "vision": data.load_and_transform_gradio_image(image, device),
        "audio": data.load_and_transform_gradio_audio(audio[0], audio[1], device),
    }

    # Instantiate model!
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    print(model)
    model.to(device)

    with torch.no_grad():
        embeddings = model(inputs)

    vision = embeddings["vision"]
    audio = embeddings["audio"]
    text = embeddings["text"]

    if vision.shape[0] == 1:
        vision_text = F.cosine_similarity(vision, text)
    else:
        vision_text = torch.softmax(vision @ text.T, dim=-1)

    if audio.shape[0] == 1:
        audio_text = F.cosine_similarity(audio, text)
    else:
        audio_text = torch.softmax(audio @ text.T, dim=-1)

    if vision.shape[0] == 1 and audio.shape[0] == 1:
        vision_audio = F.cosine_similarity(vision, audio)
    else:
        vision_audio = torch.softmax(vision @ audio.T, dim=-1)

    return f"Vision x Text: {vision_text}\nAudio x Text: {audio_text}\nVision x Audio: {vision_audio}"


def gpt_chat(context, prompt, examples=None):
    # TODO: examples converts tuples into gpt dict format
    return gpt_text(prompt, system=context)


def gpt_text(
    prompt: Union[str, List[Dict[str, str]]] = None,
    system=None,
    model="gpt-3.5-turbo",
    temperature: float = 0.6,
    max_tokens: int = 32,
    stop: List[str] = ["\n"],
):
    import openai
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif prompt is None:
        prompt = []
    if system is not None:
        prompt = [{"role": "system", "content": system}] + prompt
    log.debug(f"Function call to GPT {model}: \n {prompt}")
    response = openai.ChatCompletion.create(
        messages=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    return response["choices"][0]["message"]["content"]


def gpt_emoji(prompt):
    try:
        emoji = gpt_text(
            prompt=prompt,
            system=" ".join(
                [
                    "Respond with a single emoji based on the user prompt.",
                    "Respond with only basic original emoji.",
                    "Respond with only one emoji.",
                    "Do not explain, respond with the single emoji only.",
                ]
            ),
            temperature=0.3,
        )
    except Exception:
        emoji = "👻"
    return emoji


def gpt_color():
    try:
        color_name = gpt_text(
            system=" ".join(
                [
                    "You generate unique and interesting colors for a crayon set.",
                    "Crayon color names are only a few words.",
                    "Respond with the colors only: no extra text or explanations.",
                ]
            ),
            temperature=0.99,
        )
        rgb = gpt_text(
            prompt=color_name,
            system=" ".join(
                [
                    "You generate RGB color tuples for digital art based on word descriptions.",
                    "Respond with three integers in the range 0 to 255 representing R, G, and B.",
                    "The three integers should be separated by commas, without spaces.",
                    "Respond with the colors only: no extra text or explanations.",
                ]
            ),
            temperature=0.1,
        )
        rgb = rgb.split(",")
        assert len(rgb) == 3
        rgb = tuple([int(x) for x in rgb])
        assert all([0 <= x <= 256 for x in rgb])
    except Exception:
        color_name = "black"
        rgb = (0, 0, 0)
    return rgb, color_name


def gpt_image(
    prompt: str,
    n: int = 1,
    image_size="512x512",
):
    import openai
    log.debug(f"Image call to GPT with: \n {prompt}")
    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size=image_size,
    )
    img_url = response["data"][0]["url"]
    image = Image.open(BytesIO(requests.get(img_url).content))
    # Output path for original image
    image_name = uuid.uuid4()
    image_path = os.path.join(DATA_DIR, f"{image_name}.png")
    image.save(image_path)
    return image_path


# Define the main GradIO UI
with gr.Blocks() as demo:
    gr.Markdown(
"""
# fren 🤖

"""
    )
    log.info("Starting GradIO Frontend ...")
    texts_references = gr.State(value="")
    gr_button_clear_gpu = gr.Button(value="Clear GPU")
    gr_button_clear_gpu.click(
        clear_gpu,
        inputs=[],
        outputs=[],
    )
    with gr.Tab("Texts"):
        with gr.Row():
            with gr.Column():
                gr_text_context_gpt = gr.Textbox(
                    placeholder="GPT context",
                    show_label=False,
                )
                gr_text_prompt_gpt = gr.Textbox(
                    placeholder="GPT prompt",
                    show_label=False,
                )
                gr_button_gpt = gr.Button(value="Chat")
                gr_text_output_gpt = gr.Textbox(
                    placeholder="GPT output",
                    show_label=False,
                )
                gr_button_gpt.click(
                    gpt_chat,
                    inputs=[gr_text_prompt_gpt, gr_text_context_gpt],
                    outputs=[gr_text_output_gpt],
                )
            with gr.Column():
                gr_text_context_palm = gr.Textbox(
                    placeholder="PaLM context",
                    show_label=False,
                )
                gr_text_prompt_palm = gr.Textbox(
                    placeholder="PaLM prompt",
                    show_label=False,
                )
                gr_button_palm = gr.Button(value="Chat")
                gr_text_output_palm = gr.Textbox(
                    placeholder="PaLM output",
                    show_label=False,
                )
                gr_button_palm.click(
                    palm_chat,
                    inputs=[gr_text_prompt_palm, gr_text_context_palm],
                    outputs=[gr_text_output_palm],
                )

    with gr.Tab("ImageBind"):
        gr_image = gr.Image(
            label="Image",
            image_mode="RGB",
            # value="/home/tren/Downloads/cat.png",
        )
        with gr.Column():
            gr_generate_button = gr.Button(value="Generate Image")
            gr_prompt_textbox = gr.Textbox(
                placeholder="Image Prompt",
                show_label=False,
                lines=1,
                value="",
            )
        gr_generate_button.click(
            gpt_image,
            inputs=[gr_prompt_textbox],
            outputs=[gr_image],
        )
        with gr.Column():
            gr_audio = gr.Audio(
                label="Audio",
                source="microphone",
                # format="wav",
                # type="numpy",
            )
            gr_text_input = gr.Textbox(
                placeholder="Text",
                show_label=False,
                lines=1,
                value="",
            )
            gr_bind_button = gr.Button(value="ImageBind")
        gr_text_output = gr.Textbox(
            placeholder="Output",
            show_label=False,
            value="",
        )
        gr_bind_button.click(
            imagebind,
            inputs=[gr_text_input, gr_audio, gr_image],
            outputs=[gr_text_output],
        )
            
    with gr.Tab("Keys"):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        openai_api_key_textbox.change(
            set_openai_key,
            inputs=[openai_api_key_textbox],
        )
        set_openai_key()
        with gr.Accordion(
                    label="GPT Settings",
                    open=False,
                ):
                    gr_model = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-4"],
                        label="GPT Model behind conversation",
                        value="gpt-3.5-turbo",
                    )
                    gr_max_tokens = gr.Slider(
                        minimum=1,
                        maximum=300,
                        value=50,
                        label="max tokens",
                        step=1,
                    )
                    gr_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        label="Temperature",
                    )
        huggingface_api_key_textbox = gr.Textbox(
            placeholder="Paste your HuggingFace API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        huggingface_api_key_textbox.change(
            set_huggingface_key,
            inputs=[huggingface_api_key_textbox],
        )
        set_huggingface_key()
        palm_api_key_textbox = gr.Textbox(
            placeholder="Paste your Palm API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        palm_api_key_textbox.change(
            set_palm_key,
            inputs=[palm_api_key_textbox],
        )
        set_palm_key()

    gr.HTML(
        """
        <center>
        Author: <a href="https://youtube.com/@hu-po">Hu Po</a>
        GitHub: <a href="https://github.com/hu-po/fren">fren</a>
        <br>
        <a href="https://huggingface.co/spaces/hu-po/fren?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        </center>
        """
    )

if __name__ == "__main__":
    demo.launch()
