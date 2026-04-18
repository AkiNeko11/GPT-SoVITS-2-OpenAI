# GPT-SoVITS-2-OpenAI

这是一个为 GPT-SoVITS 编写的 OpenAI 风格语音合成 API 插件。

配合我的另一个项目 [myGPT-SoVITS](https://github.com/AkiNeko11/myGPT-SoVITS) 使用。  
使用时，只需将 `app.py` 和`config`文件放置在 `plugin` 目录下即可。

该插件实现了与 OpenAI 语音合成 API 类似的功能，便于其他项目集成调用。  
同时通过 Flask-CORS 支持浏览器的跨域请求访问。


| 端点                   | 方法   | 功能                               |
|-----------------------|--------|------------------------------------|
| /v1/audio/speech      | POST   | 核心 TTS 端点，兼容 OpenAI 格式    |
| /v1/models            | GET    | 列出可用声音模型                   |
| /v1/voices            | GET    | 列出可用声音                       |
| /v1/config/reload     | POST   | 热重载 config.yaml 配置            |
| /health               | GET    | 健康检查                           |