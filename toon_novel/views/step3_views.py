from flask import Flask, Blueprint, render_template, send_file, request, jsonify

bp = Blueprint('step3', __name__, url_prefix='/step3')

@bp.route('/')
def create_ilst():
    return 0