from flask import Flask, render_template, url_for

from views import step1_views, step2_views, step3_views, text_info_views, text_translation_views


app = Flask(__name__)

app.register_blueprint(step1_views.bp)
app.register_blueprint(step2_views.bp)
app.register_blueprint(step3_views.bp)

app.register_blueprint(text_info_views.bp)
app.register_blueprint(text_translation_views.bp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)