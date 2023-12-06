# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import os
from argparse import ArgumentParser

import gradio as gr
import mdtex2html
import piexif
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from diffusers import AutoPipelineForText2Image
import torch
import json
import time
import datetime
from file_server import start_server, get_local_ip

DEFAULT_CKPT_PATH = 'hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt'
OUTPUT_IMAGES_DIR = "output_images"
OUTPUT_HTML_DIR = "output_html"


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
    parser.add_argument("--file-server-port", type=int, default=8001,
                        help="file server port.")
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


def _save_image2html(image, query, prompt):
    # 将文本信息编码为 JSON 并保存到 EXIF
    exif_dict = {"0th": {}, "Exif": {}, "1st": {}, "thumbnail": None, "GPS": {}}
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json.dumps({"prompt": prompt})
    exif_bytes = piexif.dump(exif_dict)

    file_name = f"{int(time.time())}.png"
    image_path = os.path.join(OUTPUT_IMAGES_DIR, file_name)
    image.save(image_path, "PNG", exif=exif_bytes)
    # 创建 HTML 内容
    # 初始 HTML 结构


    html_start = """<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
    <title>Image and Prompt History</title></head><body><h1>Image and Prompt History</h1><ul>"""
    html_end = "</ul></body></html>"
    # 将 HTML 内容写入文件
    html_file_path = os.path.join(OUTPUT_HTML_DIR, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.html")
    # 创建新的列表项
    new_list_item = f"""
        <li>
            <p>Prompt: {prompt}</p>
            <p>Input: {query}</p>
            <img src="{image_path}" alt="{image_path}" style="max-width: 100%; height: auto;">
        </li>
    """

    # 读取现有的 HTML 文件
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            existing_html = file.read()
    except FileNotFoundError:
        # 如果文件不存在，创建一个新的 HTML 结构
        existing_html = html_start + html_end

    # 在列表结束标签前插入新的列表项
    updated_html = existing_html.replace(html_end, new_list_item + html_end)

    # 将更新后的 HTML 写回文件
    with open(html_file_path, 'w+', encoding='utf-8') as file:
        file.write(updated_html)

    return f"HTML content appended to {html_file_path}"


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
    def predict(_query, _chatbot, _task_history, prompt_template: str, prompt_system: str):
        print(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""
        _query = f"{prompt_template}\n{_query}"

        for response in model.chat_stream(
                tokenizer,
                _query,
                history=_task_history,
                generation_config=config,
                system=prompt_system
        ):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Qwen-Chat: {_parse_text(full_response)}")

    def draw_image(_chatbot, _task_history, num_inference_steps, args):
        if len(_task_history) == 0:
            return
        prompt = _task_history[-1][-1]
        if len(prompt) == 0:
            return
        print(f"===\n{_chatbot} \n\n{_task_history} ====\n")
        print(f"{prompt}")
        image_pil = image_pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]
        _save_image2html(image_pil, query=_chatbot[-1][0], prompt=prompt)
        return image_pil

    def regenerate(_chatbot, _task_history, prompt_system):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history, prompt_template="", prompt_system=prompt_system)

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
        file_server = f"http://{get_local_ip()}:{args.file_server_port}/"
        html_file_path = f"{datetime.datetime.now().strftime('%Y-%m-%d')}.html"
        html_fns = [fn for fn in os.listdir(OUTPUT_HTML_DIR) if fn.endswith(".html")]

        gr.Markdown(f'<a href="{file_server}{html_file_path}" target="_blank">{html_file_path}</a>')
        for fn in html_fns:
            if fn == html_file_path:
                continue
            gr.Markdown(f'<a href="{file_server}{fn} target="_blank"">{fn}</a>')
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                image = gr.Image(type="pil")
                query = gr.Textbox(lines=4, label='Input')

            with gr.Column(scale=1, min_width=600):
                with gr.Tab(label="Qwen"):
                    with gr.Row():
                        prompt_system_radio = gr.Radio(
                            ["中英文翻译", "文言文", "画家", "剧情"],
                            label='角色',
                            info="根据输入选择合适的角色"
                        )
                        prompt_system = gr.Textbox(
                            lines=1,
                            label='System Template',
                            value="你擅长翻译中文到英语。"
                        )

                    prompt_template = gr.Textbox(
                        lines=1,
                        label='Prompt Template',
                        value="必须使用英语根据主题描述一副画面:"
                    )
                    chatbot = gr.Chatbot(label='Qwen-Chat', elem_classes="control-height")

                with gr.Tab(label="Config"):
                    with gr.Row():
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Temperature")
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Top-p")
                        top_k = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top-k")
                        max_new_tokens = gr.Slider(minimum=1, maximum=1024, step=1, value=100, label="Max New Tokens")
                    with gr.Row():
                        num_inference_steps = gr.Slider(minimum=1, maximum=60, step=1, value=4, label="Image Steps")
                task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            image_btn = gr.Button("🎨 Image (生成)")

        PROMPT_SYSTEM_DICT = {
            "中英文翻译": "你擅长翻译中文到英语。",
            "文言文": "你擅长文言文翻译为英语。",
            "画家": "你是绘画大师，擅长描绘画面细节。",
            "剧情": "你是剧作家，擅长创作连续的漫画脚本。"
        }
        prompt_system_radio.change(lambda val: (PROMPT_SYSTEM_DICT[val]),
                                   inputs=[prompt_system_radio], outputs=[prompt_system])
        temperature.change(lambda val: config.update(temperature=val), inputs=[temperature], outputs=[])
        top_k.change(lambda val: config.update(top_k=val), inputs=[top_k], outputs=[])
        top_p.change(lambda val: config.update(top_p=val), inputs=[top_p], outputs=[])
        max_new_tokens.change(
            lambda val: config.update(max_new_tokens=val),
            inputs=[max_new_tokens],
            outputs=[],
        )

        submit_btn.click(predict, [query, chatbot, task_history, prompt_template, prompt_system], [chatbot],
                         show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        image_btn.click(draw_image, [chatbot, task_history, num_inference_steps], outputs=[image],
                        show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history, prompt_system], [chatbot], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    start_server(server_port=args.file_server_port)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)
    model, tokenizer, config = _load_model_tokenizer(args)
    pipe = _load_sdxl_turbo()
    _launch_demo(args, pipe, model, tokenizer, config)


if __name__ == '__main__':
    main()
