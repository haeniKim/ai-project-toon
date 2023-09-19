from flask import Flask, Blueprint, render_template, send_file, request, jsonify
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import deepl
import time

bp = Blueprint('step1', __name__, url_prefix='/step1')

# hugging_token = ''

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# # Deepl 번역 API 설정
# translator = deepl.Translator("")

# pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
#                                              torch_dtype=torch.float32, 
#                                              use_safetensors=True, 
#                                              variant="fp16").to(device)

# # 서버 아닐 시
# # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
# #                                                revision='fp16',
# #                                                torch_dtype=torch.float32,
# #                                                use_auth_token=hugging_token
# #                                                ).to(device)

# @bp.route('/', methods=['POST', 'GET'])
# def create_basic():
#     if request.method == 'POST':
#         # POST request received

#         # Extract data from the form
#         title = request.form['title']
#         name = request.form['name']
#         desc = request.form['description']
#         genre = request.form.getlist("genre")

#         # Translate the text using the Deepl API
#         result = translator.translate_text(f"{name} : {desc}, 웹툰그림체, 장르는 {genre}, 정면 응시, 상반신, 배경 없음", target_lang="en-us")

#         # Set up the prompt
#         prompt = result.text

#         # Generate the image
#         image = pipeline(prompt).images[0]

#         # Generate a unique image filename
#         timestamp = int(time.time())
#         image_filename = f'static/image/{title}_{name}_{timestamp}.png'  # Save to the static/image folder

#         # Save the image to the specified path
#         image.save(image_filename, format='PNG')

#         # Return the URL of the saved image
#         image_url = f"/{image_filename}"  # Assuming the app is hosted at the root URL

#         return render_template("step1_2.html", image_url=image_url)
    
#     # Render the HTML form for GET requests
#     return render_template("step1.html")
