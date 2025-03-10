from openai import OpenAI

# client = OpenAI(api_key='empty', base_url='http://powerai.cc:8001/v1')
# model_name = client.models.list().data[0].id
# response = client.chat.completions.create(
#     model=model_name,
#     messages=[{
#         'role':
#         'user',
#         'content': [{
#             'type': 'text',
#             'text': '图片里是什么？',
#         }, {
#             'type': 'image_url',
#             'image_url': {
#                 'url':
#                 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
#             },
#         }],
#     }],
#     temperature=0.8,
#     top_p=0.8)
# print(response)

# client = OpenAI(api_key='f5565670-0583-41f5-a562-d8e770522bd7', base_url='https://ark.cn-beijing.volces.com/api/v3/')
client = OpenAI(api_key='empty', base_url='https://powerai.cc:8001/v1/')
# print(f'client.models.list(): "{client.models.list()}"')
# model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model='qwq-32b',
    # model='deepseek-v3-241226',
    messages=[
      {"role": "system","content": "你是人工智能助手."},
      {"role": "user","content": "你是谁？"}
    ],
    temperature=0.6,
)
print(response)

