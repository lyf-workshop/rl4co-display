# config.example.py
# 配置文件模板 - 请复制为 config.py 并修改实际值

class Config:
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'  # 替换为你的 MySQL 用户名
    MYSQL_PASSWORD = 'your_password_here'  # 请修改为你的 MySQL 密码
    MYSQL_DB = 'flaskdemo_user'

    # ──────────────────────────────────────────────────────────
    # 云端 LLM 服务商配置
    # 支持任意兼容 OpenAI API 格式的服务商
    # 填写 api_key 后该服务商才会出现在前端来源列表中
    # ──────────────────────────────────────────────────────────
    LLM_PROVIDERS = [
        {
            'id':            'deepseek',
            'name':          'DeepSeek',
            'base_url':      'https://api.deepseek.com/v1',
            'api_key':       '',          # 填入你的 DeepSeek API Key
            'models':        ['deepseek-chat', 'deepseek-reasoner'],
            'default_model': 'deepseek-chat',
        },
        # {
        #     'id':            'qwen',
        #     'name':          '通义千问',
        #     'base_url':      'https://dashscope.aliyuncs.com/compatible-mode/v1',
        #     'api_key':       '',
        #     'models':        ['qwen-plus', 'qwen-turbo', 'qwen-max'],
        #     'default_model': 'qwen-plus',
        # },
        # {
        #     'id':            'moonshot',
        #     'name':          'Moonshot (Kimi)',
        #     'base_url':      'https://api.moonshot.cn/v1',
        #     'api_key':       '',
        #     'models':        ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k'],
        #     'default_model': 'moonshot-v1-8k',
        # },
        # {
        #     'id':            'siliconflow',
        #     'name':          '硅基流动',
        #     'base_url':      'https://api.siliconflow.cn/v1',
        #     'api_key':       '',
        #     'models':        ['deepseek-ai/DeepSeek-V3', 'Qwen/Qwen2.5-72B-Instruct'],
        #     'default_model': 'deepseek-ai/DeepSeek-V3',
        # },
        # {
        #     'id':            'openai',
        #     'name':          'OpenAI',
        #     'base_url':      'https://api.openai.com/v1',
        #     'api_key':       '',
        #     'models':        ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
        #     'default_model': 'gpt-4o-mini',
        # },
    ]
