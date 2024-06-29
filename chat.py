from openai import OpenAI
import httpx


def create_insecure_client():
    """
    创建一个忽略证书验证的 httpx.Client。
    """
    client = httpx.Client(verify=False)
    return client


def chat(user_input):
    client = OpenAI(
        # #将这里换成你在aihubmix api keys拿到的密钥
        api_key="XXX",
        # 这里将官方的接口访问地址，替换成aihubmix的入口地址
        base_url="https://aihubmix.com/v1",
        http_client=create_insecure_client()
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content



# 以下是之前的聊天程序代码
def get_user_input():
    return input("你: ")



print("AI 聊天机器人已启动，输入 '退出' 结束对话。")
while True:
    user_input = get_user_input()
    if user_input.lower() == '退出':
        print("会话结束。")
        break
    ai_response = chat(user_input)
    print(f"AI: {ai_response}")


