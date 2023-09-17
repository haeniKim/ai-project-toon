from flask import Flask, Blueprint, render_template, send_file, request, jsonify
import deepl
from flask import Blueprint, Flask, jsonify, request
import openai
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import os

bp = Blueprint('step3', __name__, url_prefix='/step3')

# api
translator = deepl.Translator('')  ### 삭제
openai.api_key = ""  ### 삭제

model = "gpt-3.5-turbo"

hugging_token = '' ### 삭제

# cuda 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 번역  
def trans_ko_eng(input):
    result = translator.translate_text(input, target_lang='EN-US')
    return result

# 영어로 번역해서 이미지 출력
@bp.route('/trans_eng', methods=['POST'])
def eng_show():
    data = request.json

    genre = data.get('genre', '')
    character = data.get('character', '')
    plot = data.get('plot', '')
    input = data.get('input', '')

    trans_genre = str(trans_ko_eng(genre))
    trans_character = str(trans_ko_eng(character))
    trans_plot = str(trans_ko_eng(plot))
    trans_input = str(trans_ko_eng(input))

    # stable diffusion 프롬프트 작성
    def summarization(trans_genre, trans_character, trans_plot, trans_input):
    
        messages = [
                    {"role": "system", "content": f"Please extract the key points from {trans_plot} based on {trans_input} from {trans_character}'s perspective, you don't need to include the entire plot summary"},
                    {"role": "system", "content": f"Please give the elements separated by commas, such as verbs, subjects and {trans_genre} novel"},
            ]
    
        response = openai.ChatCompletion.create(
        model = model,
        messages=messages,
        max_tokens = 3000
    
        )

        result = response['choices'][0]['message']['content']

        return result
    
    result = summarization(trans_genre, trans_character, trans_plot, trans_input)

    # 프롬프트 결과 출력
    # return jsonify({"결과" : result})

    # # 서버 아닐 시
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
                                                   revision='fp16',
                                                   torch_dtype=torch.float32,
                                                   use_auth_token=hugging_token
                                                   ).to(device)
    
    
    # 이미지 파일 저장 디렉토리
    save_directory = 'D:/toonmaker_git/ai-project-toon/toon_novel/static/image/novel'

    # 이미지 파일 저장 번호 초기화
    image_number = 1

    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16").to(device)
    # pipe = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float32,
    #     use_safetensors=True,
    #     variant="fp16",
    #     max_split_size_mb=20,  # 메모리 크기에 따라 적절한 값을 선택하세요.
    # ).to(device)


    # 이미지 파일 이름 생성 및 확인
    while True:
        file_name = f'test{image_number}.png'
        file_path = os.path.join(save_directory, file_name)

        # 이미 파일이 존재하지 않는 경우에만 저장
        if not os.path.exists(file_path):
            n_steps = 40
            high_noise_frac = 0.8

            # 번역 수행
            prompt = result
            print(f'prompt : {prompt}')

            image = pipe(prompt).images[0]

            # 이미지를 파일로 저장
            image.save(file_path.replace('\\', '/')) 
            print(f"이미지를 {file_name} 파일로 저장했습니다.")
            break
        
        image_number += 1
        
    return '웹소설 삽화 생성 성공'
    


# @bp.route('/trans_eng', methods=['POST'])
# def eng_show():
#     data = request.json
#     input_text = data.get('english', '')  
#     trans_output = trans_ko_eng(input_text)
#     return jsonify({"번역한 내용": trans_output.text})

