from flask import Flask, jsonify
import json
import requests
import tensorflow as tf
import render_template
#app is the application name
app=Flask(__name__)

@app.route('/')
def home():
    model_path = "Network/"
    picture_path="Network/Test.png"
    return predictThat(picture_path,model_path)

@app.route('/Print_My_Name/<my_name>', methods=['GET'])
def print_name(my_name):
    age = request.args.get("age")
    if age is not None:
        return jsonify("you name is " + my_name+" and your age is " + str(age))
    return jsonify("you name " + my_name),200

@app.route('/Show_Meme/', methods=['GET'])
def get_meme():
    model_path = "https://drive.google.com/drive/folders/1MaFl8UBjG_gH3_bYSz2vZnd5T9PX9byD?usp=drive_link"
    response = requests.get(url)
    data = response.json()
    #print(resp.text)
    #meme_large = response["preview"][-2]
    #subreddit = response["subreddit"]
    return data["organization_teams_url"]


def predictThat(img_path,path):
  img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
  img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
  img = tf.image.convert_image_dtype(img,dtype=tf.float32,saturate=False)
    # 4. Resize to the desired size
  img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
  img = tf.transpose(img, perm=[1, 0, 2])
  max_len = 1
  dtype = tf.float32
  samples = tf.TensorArray(dtype=dtype, size=max_len, clear_after_read=False)
  for i in tf.range(max_len):
   sample = tf.cast(img, dtype=dtype)
   samples = samples.write(i, sample)
  model= keras.models.load_model(path, custom_objects={'CTCLayer': CTCLayer},compile=True)
  model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
  )
  preds = model.predict(samples.stack())
  # num_to_char is the mapping from number to char
  pred_texts, acc = decode_single_prediction(preds, num_to_char)
  import math
  print(" prediction: " + pred_texts + " acc: " + str( round((math.exp(acc)*100),2)), "%")
  pred_texts = decode_batch_predictions(preds)
  print("pred_texts     ",pred_texts)
  a = arabic_reshaper.reshape(pred_texts)
  a = get_display(a)
  return a

if __name__=="__main__":
    app.run(debug=True)