from flask import Flask, Blueprint, render_template, send_file, request, jsonify


bp = Blueprint('step2', __name__, url_prefix='/step2')




#일단은 기본 코드로(텍스트 입력 받기) -> 소연이 코드로 수정하기 + 파일 받아서
@bp.route('/', methods=['POST', 'GET'])
def txt_summ():
    
    if request.method == 'POST':
        # POST request received

        # Extract data from the form
        plot = request.form['plot']
    return 