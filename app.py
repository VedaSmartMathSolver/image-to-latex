from flask import Flask, jsonify
import pix2tex

app = Flask(__name__)

@app.route('/')
def home():
   return "Veda Smart Math Solver"

@app.route('/<path:base64String>')
def hello_name(base64String):
   #return '%s' % base64String
   result = pix2tex.pix2tex(base64String)
   return '%s' %result

if __name__ == '__main__':
   app.run(debug = True)