# Hugging Face 授权入口

SLUE Phase 1/2 与 FSC 的 Hugging Face 数据卡可能要求登录或接受数据集条款。请在浏览器完成授权，然后在运行环境中使用同一账号的访问令牌：

- 登录：https://huggingface.co/login
- 创建访问令牌：https://huggingface.co/settings/tokens
- SLUE Phase 1 数据卡：https://huggingface.co/datasets/asapp/slue
- SLUE Phase 2 数据卡：https://huggingface.co/datasets/asapp/slue-phase-2
- FSC 的 OpenDataLab 来源（当前本地下载源）：https://opendatalab.com/OpenDataLab/Fluent_Speech_Commands/tree/main

建议创建只读 token（`read`），不要把 token 写入仓库、配置文件或命令历史。登录方式：

```bash
hf auth login
# 按提示粘贴刚创建的 read token
hf auth whoami
```

旧版环境也可使用 `huggingface-cli login`。本工程不会代替用户提交 token；授权完成后，下载工具会复用 Hugging Face 的本地凭据缓存。
