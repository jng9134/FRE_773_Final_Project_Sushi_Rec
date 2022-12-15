# FRE_773_Final_Project_Sushi_Rec

Description

We plan to make a Sushi recommender system that recommends a new sushi to users who previously gave ratings to different types sushis. Our model is based off of data from [https://www.kamishima.net/sushi/](https://www.kamishima.net/sushi/) We use the LightFm package to make our model. You can find documentation at [https://making.lyst.com/lightfm/docs/home.html](https://making.lyst.com/lightfm/docs/home.html).

### Running Jupypter Notebook

* install packages from requirements.txt file (packages are requried for Flow and Flask as well)`

`pip install -r requirements.txt`

* Jupyter Notebook is inside the Sushi_Data folder
* Run SUSHI.ipynb

### Running Flow

* **Make sure your working directory is the folder Sushi_Data***
  * all csv files and python scripts are in this folder, none of the file paths will work otherwise
* Run the sushi_rec_flow.py to create a .metaflow folder and to upload metrics onto comet. Because the sushi_rec_flow.py is connected to comet, you need to have a comet api key

`COMET_API_KEY=xxx MY_PROJECT_NAME=yyy python sushi_rec_flow.py run`

### Running Flask App

* **Make sure your working directory is the folder Sushi_Data****
* Flask App will not run if you did not run the flow and have a metaflow metadata folder
* After running the flow at least one time, you can now run the flask app
* Run:

`python app.py`
