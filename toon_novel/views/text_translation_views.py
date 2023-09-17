import deepl
from flask import Blueprint, Flask, jsonify, request
import openai

app = Flask(__name__)

bp = Blueprint('eng', __name__, url_prefix='/')

translator = deepl.Translator('613bdad4-3ec0-d9aa-1d2c-30fe0c3141c3:fx')

openai.api_key = "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"

model = "gpt-3.5-turbo"

def trans_ko_eng(input):
    result = translator.translate_text(input, target_lang='EN-US')
    return result


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
                    {"role": "system", "content": f"Please extract the key points from {trans_plot} based on {trans_input}, you don't need to include the entire plot summary"},
                    {"role": "system", "content": f"Please give the elements separated by commas, such as verbs, subjects and {trans_genre} novel, from {trans_character}'s perspective"},
            ]
    
        response = openai.ChatCompletion.create(
        model = model,
        messages=messages,
        max_tokens = 3000
    
        )

        result = response['choices'][0]['message']['content']

        return result
    
    prompt = summarization(trans_genre, trans_character, trans_plot, trans_input)

    return prompt





# @bp.route('/trans_eng', methods=['POST'])
# def eng_show():
#     data = request.json
#     input_text = data.get('english', '')  
#     trans_output = trans_ko_eng(input_text)
#     return jsonify({"번역한 내용": trans_output.text})