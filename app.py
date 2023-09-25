import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)

DESCRIPTION = '''# Doodly - T2I-Adapter-SDXL **Sketch**
To try out all the [6 T2I-Adapter](https://huggingface.co/collections/TencentARC/t2i-adapter-sdxl-64fac9cbf393f30370eeb02f) released for SDXL, [click here](https://huggingface.co/spaces/TencentARC/T2I-Adapter-SDXL)
'''

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + negative


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
    )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id,
        vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
        adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)
else:
    pipe = None

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def run(
    image: PIL.Image.Image,
    prompt: str,
    negative_prompt: str,
    style_name: str = DEFAULT_STYLE_NAME,
    num_steps: int = 25,
    guidance_scale: float = 5,
    adapter_conditioning_scale: float = 0.8,
    adapter_conditioning_factor: float = 0.8,
    seed: int = 0,
    progress=gr.Progress(track_tqdm=True),
) -> PIL.Image.Image:
    image = image.convert("RGB")
    image = TF.to_tensor(image) > 0.5
    image = TF.to_pil_image(image.to(torch.float32))

    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        adapter_conditioning_scale=adapter_conditioning_scale,
        adapter_conditioning_factor=adapter_conditioning_factor,
    ).images[0]
    return out


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION, elem_id="description")
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )

    with gr.Row():
        with gr.Column():
            with gr.Group():
                image = gr.Image(
                    source="canvas",
                    tool="sketch",
                    type="pil",
                    image_mode="L",
                    invert_colors=True,
                    shape=(1024, 1024),
                    brush_radius=4,
                    height=440,
                )
                prompt = gr.Textbox(label="Что вы хотите, чтобы ИИ генерировал?")
                style = gr.Dropdown(label="Style", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
                run_button = gr.Button("Run")
            with gr.Accordion("Advanced options", open=False):
                negative_prompt = gr.Textbox(
                    label="Что вы не хотите, чтобы ИИ генерировал?",
                    value=" extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured",
                )
                num_steps = gr.Slider(
                    label="Количество итераций",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=25,
                )
                guidance_scale = gr.Slider(
                    label="Шкала расхождения",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                adapter_conditioning_scale = gr.Slider(
                    label="Адаптер конфигурации шкалы",
                    minimum=0.5,
                    maximum=1,
                    step=0.1,
                    value=0.8,
                )
                adapter_conditioning_factor = gr.Slider(
                    label="Коэффициент стабилизации адаптера",
                    info="Доля временных шагов, для которых следует применять адаптер",
                    minimum=0.5,
                    maximum=1,
                    step=0.1,
                    value=0.8,
                )
                seed = gr.Slider(
                    label="Точка старта функции",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Рандомайзер точек старта", value=True)
        with gr.Column():
            result = gr.Image(label="Результат", height=400)

    inputs = [
        image,
        prompt,
        negative_prompt,
        style,
        num_steps,
        guidance_scale,
        adapter_conditioning_scale,
        adapter_conditioning_factor,
        seed,
    ]
    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=run,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=run,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=run,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
demo.queue(max_size=20).launch(debug=True, max_threads=True, share=True, inbrowser=True)
