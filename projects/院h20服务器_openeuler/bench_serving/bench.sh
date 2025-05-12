source /home/tutu/miniconda3/etc/profile.d/conda.sh
conda activate sgl45
python -m sglang.bench_serving --backend vllm \
--dataset-name random \
--dataset-path /home/tutu/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
--random-input 128 \
--random-output 1024 \
--num-prompts 1  \
--host 172.19.80.137 --port 28002 \
--tokenizer /home/tutu/models/QwQ-32B \
--output-file "sglang_bench.txt"

# 1)如果是vllm开的api，就要指定--backend vllm
# 2)dataset下载地址：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
#     不指定--dataset-path时，运行后会下载：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
# 3)json文件内容格式：
[
  {
    "id": "QWJhYvA_0",
    "conversations": [
      {
        "from": "human",
        "value": "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."
      },
      {
        "from": "gpt",
        "value": "Here are the main ideas of Jeff Walker's Product Launch Formula that can be applied by a growth marketing agency for their clients:\n\n1. Identify the target audience and their needs: Understand the ideal customer for the product or service, and create a messaging that resonates with them.\n2. Pre-launch: Build anticipation and excitement for the launch by creating buzz, gathering testimonials and case studies, and using social media to create awareness.\n3. Launch: Use a well-crafted launch sequence to maximize sales and conversions. This can include offering bonuses, creating scarcity, and using a deadline to create urgency.\n4. Post-launch: Follow up with customers, gather feedback, and continue to provide value to keep them engaged and loyal.\n5. Create a product suite: Once the initial product is successful, expand the offering by creating additional products or services that address other needs of the same audience.\n6. Continual optimization: Continually monitor and optimize the launch process and product suite to improve results.\n7. Build a community: Use the launch process to build a community of customers who are passionate about the product and can help spread the word.\n8. Use automation: Use technology and automation to streamline the launch process and improve efficiency."
      },
      {
        "from": "human",
        "value": "Summarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."
      },
      {
        "from": "gpt",
        "value": "Here are the main ideas of Brendon Burchard's Experts Academy that can be applied by a growth marketing agency for their clients:\n\n1. Identify your expertise: Define your area of expertise and create a unique value proposition that differentiates you from others.\n2. Create a content strategy: Develop a content strategy that showcases your expertise and provides value to your audience. This can include blog posts, videos, podcasts, and social media content.\n3. Build your audience: Use social media, advertising, and other channels to build your audience and increase your visibility.\n4. Develop your products: Create products and services that align with your expertise and meet the needs of your audience. These can include digital products, courses, coaching, and consulting services.\n5. Create a brand: Develop a strong brand that reflects your expertise and resonates with your audience.\n6. Build authority: Establish yourself as an authority in your industry by speaking at events, publishing a book, or appearing on podcasts or TV shows.\n7. Monetize your expertise: Develop a monetization strategy that leverages your expertise and products to generate revenue.\n8. Build a team: As your business grows, build a team of experts to help you scale your business and provide excellent service to your clients.\n9. Continual improvement: Continually improve your products, services, and marketing strategies to stay ahead of the competition and provide the best possible experience for your clients.\n10. Focus on impact: Ultimately, focus on making a positive impact in the lives of your clients and using your expertise to make a difference in the world."
      },
      {
        "from": "human",
        "value": "What are the mental triggers in Jeff Walker's Product Launch Formula and \"Launch\" book?"
      },
      {
        "from": "gpt",
        "value": "Here are the mental triggers described in Jeff Walker's Product Launch Formula and \"Launch\" book:\n\n1. Scarcity: Creating a sense of urgency by limiting the availability of the product or service being launched.\n2. Authority: Establishing oneself as an expert or authority in the industry to build credibility and trust with the audience.\n3. Social proof: Using testimonials, case studies, and other forms of social proof to demonstrate the effectiveness and value of the product or service.\n4. Reciprocity: Offering something of value to the audience before asking for a sale or commitment.\n5. Likeability: Building a personal connection with the audience and creating a likable persona to increase the chances of a sale.\n6. Anticipation: Building anticipation and excitement leading up to the launch to generate buzz and increase interest in the product or service.\n7. Exclusivity: Creating a feeling of exclusivity by offering early access or special deals to a select group of people.\n8. Belonging: Creating a sense of community or belonging among the audience to increase engagement and loyalty to the brand.\n9. Storytelling: Using storytelling to connect with the audience on an emotional level and make the product or service more relatable and memorable.\n\nBy leveraging these mental triggers, businesses can create an effective product launch strategy that resonates with their target audience and drives sales and growth."
      },
    ]
  }
]