from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from .utils import *
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
		
		#Prepare the feature vector for prediction

        title = result['title']
        body = result['body']
        title_body=title+" "+body
        supervised_tags=predict_S(title_body)
        unsupervised_tags=predict_U(title_body)
		
        return render_template('result.html',title=title, body=body, supervised_tags=supervised_tags, unsupervised_tags=unsupervised_tags)

    
if __name__ == '__main__':
	app.run()
