**English Version:**

---

# Qwen-SDXL-Turbo

Qwen creates prompts for SDXL.

## Model

Download the model at [this link](https://huggingface.co/hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt).

## Installation

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Use the following command-line arguments to operate this script:

- `-c`, `--checkpoint-path`: Specifies the checkpoint name or path, defaulting to `hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt`.
- `-x`, `--sdxl-path`: Defines the SDXL Checkpoint name or path. Options are:
  - `Lykon/dreamshaper-xl-turbo`, recommended `num_inference_steps` are between 6 and 16 for enhanced image quality.
  - `stabilityai/sdxl-turbo`, suggested `num_inference_steps` are between 1 and 4 for quicker processing.
- `--share`: Generates a publicly shareable link for the interface. Default is set to `False`.
- `--inbrowser`: Opens the interface in a new tab of the default browser automatically. Default is `False`.
- `--server-port`: The port number for the demo server. Default is `8000`.
- `--server-name`: The name for the demo server. Default is `0.0.0.0`.
- `--file-server-port`: Port number for the file server. Default is `8001`.

## Example

Run the web demo with the following command:

```bash
python web_demo.py
```

![demo.jpg](demo.jpg)



**Chinese:**



# Qwen-SDXL-Turbo

Qwen 用于为 SDXL 创建Prompt。

## 模型

在[此链接](https://huggingface.co/hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt)下载模型。

## 安装

使用以下命令安装所需包：

```bash
pip install -r requirements.txt
```

## 使用方法

使用以下命令行参数操作此脚本：

- `-c`, `--checkpoint-path`：指定检查点名称或路径，默认为 `hahahafofo/Qwen-1_8B-Stable-Diffusion-Prompt`。
- `-x`, `--sdxl-path`：定义 SDXL 检查点名称或路径。选项包括：
  - `Lykon/dreamshaper-xl-turbo`，建议 `num_inference_steps` 在 6 到 16 之间，以获得更好的图像质量。
  - `stabilityai/sdxl-turbo`，建议 `num_inference_steps` 在 1 到 4 之间，以实现更快的处理速度。
- `--share`：生成接口的公开共享链接。默认设置为 `False`。
- `--inbrowser`：自动在默认浏览器的新标签页中打开接口。默认为 `False`。
- `--server-port`：演示服务器的端口号。默认为 `8000`。
- `--server-name`：演示服务器的名称。默认为 `0.0.0.0`。
- `--file-server-port`：文件服务器的端口号。默认为 `8001`。

## 示例

使用以下命令运行：

```bash
python web_demo.py
```

![demo.jpg](demo.jpg)

---
