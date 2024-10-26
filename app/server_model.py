import os
from flask import Flask, request, jsonify, make_response
import requests
import json
import prediction.py
import load_models.py


model_assortment, optimizer_assortment, model_promo, optimizer_promo, model_exist, optimizer_exist = None, None, None, None, None, None

app = Flask(__name__)


@app.route("/")
def status():
    global model_assortment, optimizer_assortment, model_promo, optimizer_promo, model_exist, optimizer_exist
    model_assortment, optimizer_assortment, model_promo, optimizer_promo, model_exist, optimizer_exist = load_models()
    return jsonify({"status": "ok"})


@app.route('/recognition',methods=['GET', 'POST'])
def upload_file():

    input_json = request.get_json(force = True)
    img_url = input_json['image_url']
    request_id = input_json['request_id']
    r = requests.get(img_url)
    if r.status_code == 200:
        with open('/recognition/input_photos/' + str(request_id) + '__' + '.jpg', 'wb') as f:
            f.write(r.content)
            
        output_json = prediction(model_assortment, optimizer_assortment, model_promo, optimizer_promo, model_exist, optimizer_exist)
        os.remove('/recognition/input_photos/' + str(request_id) + '__' + '.jpg')
        return jsonify(output_json)
    else:
        return make_response(jsonify({"message": "Image_url must be contain link with url of photo."}), 400)

    
