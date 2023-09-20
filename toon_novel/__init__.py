from flask import Flask, render_template, url_for, request

from views import step1_views, step2_views, step3_views

app = Flask(__name__)

app.register_blueprint(step1_views.bp)
app.register_blueprint(step2_views.bp)
app.register_blueprint(step3_views.bp)

if __name__ == "__main__":
    app.run(host='localhost', port=5001, debug=True)