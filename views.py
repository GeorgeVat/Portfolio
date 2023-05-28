from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from Toxicity.transformersEncoding import *



views = Blueprint(__name__, 'views')

@views.route('/')
def home():
    return render_template("index.html")


@views.route('/toxicity')
def toxicity():
    return render_template("toxicity.html")

@views.route('/process_string', methods=['POST'])
def process_string():
    input_string = request.form.get('input_string')
    encoder = prepare_data(input_string, tokenizer)
    output1, output2 = make_prediction(encoder, model, classes=['Non-Toxic', 'Toxic'])

    
    return render_template('toxicity.html', result1=output1, result2=output2)
