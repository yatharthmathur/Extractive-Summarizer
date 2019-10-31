# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:44:50 2019

@author: Vinay Valson
"""

from flask import Flask,render_template,url_for,request,redirect,flash,session
from flask_bootstrap import Bootstrap
import matplotlib.pyplot as plt
import os
from werkzeug import secure_filename
from extractive_summarization import summarize


application = app = Flask(__name__)
Bootstrap(app)

@app.route('/',methods=['POST','GET'])
def index():
	textprint = " NO RELEVENT TEXT ADDED !!"
	if request.method == "POST":
		Textarea = request.form['Textarea']
		newtext = summarize(Textarea,33)
		return render_template('index.html',textprint = Textarea,newprinttext = newtext)
	else:
		return render_template('index.html')

if __name__ == '__main__':
	app.debug = True
	app.secret_key='12345'
	app.run()
