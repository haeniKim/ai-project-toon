import deepl
from flask import Blueprint, Flask, jsonify, request
import openai
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

app = Flask(__name__)

bp = Blueprint('eng', __name__, url_prefix='/')

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

    return jsonify({"결과": result,
                    "값 타입" : type(result)})

    #  서버 pc
    # pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
    #                                          torch_dtype=torch.float32, 
    #                                          use_safetensors=True, 
    #                                          variant="fp16").to(device)

    # # 서버 아닐 시
    # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
    #                                                revision='fp16',
    #                                                torch_dtype=torch.float32,
    #                                                use_auth_token=hugging_token
    #                                                ).to(device)
    
    # prompt = result

    # image = pipe(prompt).images[0]

    # return image



# @bp.route('/trans_eng', methods=['POST'])
# def eng_show():
#     data = request.json
#     input_text = data.get('english', '')  
#     trans_output = trans_ko_eng(input_text)
#     return jsonify({"번역한 내용": trans_output.text})