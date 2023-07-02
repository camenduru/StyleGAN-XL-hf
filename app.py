#!/usr/bin/env python

from __future__ import annotations

import json

import gradio as gr
import numpy as np

from model import Model

DESCRIPTION = '# [StyleGAN-XL](https://github.com/autonomousvision/stylegan_xl)'


def update_class_index(name: str) -> dict:
    if 'imagenet' in name:
        return gr.Slider.update(maximum=999, visible=True)
    elif 'cifar' in name:
        return gr.Slider.update(maximum=9, visible=True)
    else:
        return gr.Slider.update(visible=False)


def get_sample_image_url(name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/StyleGAN-XL/resolve/main/samples'
    return f'{sample_image_dir}/{name}.jpg'


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    if name == 'imagenet':
        size = 128
        class_index = '0-999'
        seed = '0'
    elif name == 'cifar10':
        size = 32
        class_index = '0-9'
        seed = '0-9'
    elif name == 'ffhq':
        size = 256
        class_index = 'N/A'
        seed = '0-99'
    elif name == 'pokemon':
        size = 256
        class_index = 'N/A'
        seed = '0-99'
    else:
        raise ValueError

    return f'''
    - size: {size}x{size}
    - class_index: {class_index}
    - seed: {seed}
    - truncation: 0.7
    ![sample images]({url})'''


def load_class_names(name: str) -> list[str]:
    with open(f'labels/{name}_classes.json') as f:
        names = json.load(f)
    return names


def get_class_name_df(name: str) -> list:
    names = load_class_names(name)
    return list(map(list, enumerate(names)))  # type: ignore


IMAGENET_NAMES = load_class_names('imagenet')
CIFAR10_NAMES = load_class_names('cifar10')


def update_class_name(model_name: str, index: int) -> dict:
    if 'imagenet' in model_name:
        if index < len(IMAGENET_NAMES):
            value = IMAGENET_NAMES[index]
        else:
            value = '-'
        return gr.Textbox.update(value=value, visible=True)
    elif 'cifar' in model_name:
        if index < len(CIFAR10_NAMES):
            value = CIFAR10_NAMES[index]
        else:
            value = '-'
        return gr.Textbox.update(value=value, visible=True)
    else:
        return gr.Textbox.update(visible=False)


model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('App'):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        model_name = gr.Dropdown(model.MODEL_NAMES,
                                                 value=model.MODEL_NAMES[3],
                                                 label='Model')
                        seed = gr.Slider(0,
                                         np.iinfo(np.uint32).max,
                                         step=1,
                                         value=0,
                                         label='Seed')
                        psi = gr.Slider(0,
                                        2,
                                        step=0.05,
                                        value=0.7,
                                        label='Truncation psi')
                        class_index = gr.Slider(0,
                                                999,
                                                step=1,
                                                value=83,
                                                label='Class Index')
                        class_name = gr.Textbox(
                            value=IMAGENET_NAMES[class_index.value],
                            label='Class Label',
                            interactive=False)
                        tx = gr.Slider(-1,
                                       1,
                                       step=0.05,
                                       value=0,
                                       label='Translate X')
                        ty = gr.Slider(-1,
                                       1,
                                       step=0.05,
                                       value=0,
                                       label='Translate Y')
                        angle = gr.Slider(-180,
                                          180,
                                          step=5,
                                          value=0,
                                          label='Angle')
                        run_button = gr.Button('Run')
                with gr.Column():
                    result = gr.Image(label='Result', elem_id='result')

        with gr.TabItem('Sample Images'):
            with gr.Row():
                model_name2 = gr.Dropdown([
                    'imagenet',
                    'cifar10',
                    'ffhq',
                    'pokemon',
                ],
                                          value='imagenet',
                                          label='Model')
            with gr.Row():
                text = get_sample_image_markdown(model_name2.value)
                sample_images = gr.Markdown(text)

        with gr.TabItem('Class Names'):
            with gr.Row():
                dataset_name = gr.Dropdown([
                    'imagenet',
                    'cifar10',
                ],
                                           value='imagenet',
                                           label='Dataset')
            with gr.Row():
                df = get_class_name_df('imagenet')
                class_names = gr.Dataframe(df,
                                           col_count=2,
                                           headers=['Class Index', 'Label'],
                                           interactive=False)

    model_name.change(fn=model.set_model, inputs=model_name, outputs=None)
    model_name.change(fn=update_class_index,
                      inputs=model_name,
                      outputs=class_index)
    model_name.change(fn=update_class_name,
                      inputs=[
                          model_name,
                          class_index,
                      ],
                      outputs=class_name)
    class_index.change(fn=update_class_name,
                       inputs=[
                           model_name,
                           class_index,
                       ],
                       outputs=class_name)
    run_button.click(fn=model.set_model_and_generate_image,
                     inputs=[
                         model_name,
                         seed,
                         psi,
                         class_index,
                         tx,
                         ty,
                         angle,
                     ],
                     outputs=result)
    model_name2.change(fn=get_sample_image_markdown,
                       inputs=model_name2,
                       outputs=sample_images)
    dataset_name.change(fn=get_class_name_df,
                        inputs=dataset_name,
                        outputs=class_names)

demo.queue(max_size=10).launch()
