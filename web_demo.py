# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import os
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from diffusers import AutoPipelineForText2Image
import torch

DEFAULT_CKPT_PATH = './output_qwen'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,

    )

    return model, tokenizer, config


def _load_sdxl_turbo():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to("cuda")
    return pipe


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _launch_demo(args, image_pipe, model, tokenizer, config):
    def predict(_query, _chatbot, _task_history, prompt_template):
        print(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""
        _query = f"{prompt_template}\n{_query}"

        for response in model.chat_stream(tokenizer, _query, history=_task_history, generation_config=config):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Qwen-Chat: {_parse_text(full_response)}")

    def draw_image(_chatbot, _task_history):
        if len(_task_history) == 0:
            return
        prompt = _task_history[-1][-1]
        if len(prompt) == 0:
            return
        print(f"===\n{_chatbot} \n\n{_task_history} ====\n")
        print(f"{prompt}")
        return image_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    def regenerate(_chatbot, _task_history, prompt_template):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return _chatbot

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                image = gr.Image(type="pil")
            with gr.Column(scale=1, min_width=600):
                with gr.Row():
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.9, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.9, label="Top-p")
                    top_k = gr.Slider(minimum=0, maximum=100, step=1, default=0, label="Top-k")
                    max_new_tokens = gr.Slider(minimum=1, maximum=1024, step=1, default=512, label="Max New Tokens")
                    prompt_template = gr.Textbox(
                        lines=1,
                        label='Prompt Template',
                        default="你是绘画大师，必须使用英语根据主题描述一副画面:"
                    )
                chatbot = gr.Chatbot(label='Qwen-Chat', elem_classes="control-height")
                query = gr.Textbox(lines=4, label='Input')
                task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            image_btn = gr.Button("🚀 Image (生成)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        temperature.change(
            lambda val: config.update({"temperature": val}),
            inputs=[temperature],
            outputs=[temperature],
        )
        top_k.change(
            lambda val: config.update({"top_k": val}),
            inputs=[top_k],
            outputs=[top_k],
        )
        top_p.change(
            lambda val: config.update({"top_p": val}),
            inputs=[top_p],
            outputs=[top_p],
        )
        max_new_tokens.change(
            lambda val: config.update({"max_new_tokens": val}),
            inputs=[max_new_tokens],
            outputs=[max_new_tokens],
        )

        submit_btn.click(predict, [query, chatbot, task_history, prompt_template], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        image_btn.click(draw_image, [chatbot, task_history], outputs=[image], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history, prompt_template], [chatbot], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)
    pipe = _load_sdxl_turbo()
    _launch_demo(args, pipe, model, tokenizer, config)


if __name__ == '__main__':
    main()
