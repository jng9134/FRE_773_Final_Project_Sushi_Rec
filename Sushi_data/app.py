"""
    This script runs a small Flask app that displays a simple web form for users to insert some input value
    and retrieve predictions.

    Inspired by: https://medium.com/shapeai/deploying-flask-application-with-ml-models-on-aws-ec2-instance-3b9a1cec5e13
"""
import metaflow
from flask import Flask, render_template, request, jsonify
import numpy as np
from metaflow import Flow
from metaflow import get_metadata, metadata
import pandas as pd
#### THIS IS GLOBAL, SO OBJECTS LIKE THE MODEL CAN BE RE-USED ACROSS REQUESTS ####

FLOW_NAME = 'sushi_rec_flow' # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
# metadata('../src')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())

def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.model
top_ten = latest_run.data.top_ten

# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/',methods=['POST','GET'])
def main():

  # on GET we display the page  
  if request.method=='GET':
    return render_template('index.html', project=FLOW_NAME)
  # on POST we make a prediction over the input text supplied by the user
  if request.method=='POST':
    # debug
    # print(request.form.keys())
    user_x = request.form['user_x']
    n_items = request.form['n_items']


    sushi_id = list(range(0,100))
    sushi_ratings_df = pd.read_csv("sushi_ratings_data.csv")
    df = sushi_ratings_df.drop(columns=['user_id'])
    col_names = (list(df.columns))
    sushi_names = dict(zip(sushi_id, col_names))

    sushi_df = pd.read_csv("sushi_features.csv")
    sushi_id = sushi_df[['item_ID', 'name']]
    res = pd.Series(sushi_id['name'],index=sushi_id['item_ID']).to_dict()
    list_ = []

    sushi_link = "https://www.thesushigeek.com/search?q="

    if(int(user_x) >= 5000):
      print("here")
      predictions = top_ten
      counter = 0
      for x in range(int(n_items)):
        value = predictions[counter]
        sushi = sushi_names[value]
        list_.append(sushi)
        if(counter == 0):
          sushi_link = sushi_link + res[value]
        counter+=1
        # Returning the response to the client	
      
      return("Recommend the highest rated sushi for new user {}: {} Check it out here: {}".format(user_x, list_, sushi_link))
    else:
      recommedation =  list(latest_model.predict(int(user_x), np.arange(int(n_items))))
      predictions = np.flip(list(np.argsort(recommedation)))
      counter = 0
      for x in range(int(n_items)):
        value = predictions[counter]
        sushi = sushi_names[value]
        list_.append(sushi)
        if(counter == 0):
          sushi_link = sushi_link + res[value]
        counter+=1
      return("Recommendations for user {} are: {} Check it out here: {}".format(user_x, list_, sushi_link))
    

if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)