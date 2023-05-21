# Dance-Form-Recognition

## Operational Manual

Dataset :- 
For this project, we have created our own dataset of 8 different dance forms by downloading
dance videos from the internet. The dataset includes videos of the following dance forms:
Bharatanatyam, Kathak, Kuchipudi, Odissi, Manipuri, Mohiniyattam, Sattriya, and kathakali.
Then we divided videos into 8 folders.

Introduction:
The project is divided into two main parts: training the model and using the model to make
predictions. This manual will explain how to use the trained model to make predictions.
Prerequisites:
The following packages are required to run the prediction code:
- Python
- Tensorflow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Flask
You can try installing them by running the following commands in your terminal or command
prompt:
pip install opencv-python
pip install numpy
pip install matplotlib
pip install keras
pip install flask
The following files must be present:
- model.py :- includes the code for training the model
- app.py :- includes the code for making predictions using the trained model
- index.html :- include the code for user interface
- dance_classification_model.h5 :- the trained model


Instructions:


1. Ensure that you have all the necessary dataset folder and files including app.py,
model.py, and index.html in the same directory.
27
2. Open the command prompt or terminal window and navigate to the directory where the
files are saved.
3. Run the app.py file using the following command:
python app.py
4. Once the app is running, open a web browser and go to http://127.0.0.1:8000/ or
http://localhost:8000/.
5. On the web page, you will see a form with buttons for “browse video file”and
“Predict”.
6. Select the video input using the “browse video file” button .
7. Click on the "Predict" button to predict the output.
8. The predicted dance style will be displayed below the "Predict" button


Code Overview:


The app.py script uses the pre-trained dance_classification_model.h5 model to make
predictions about a dance video. The model predicts the dance style by processing each frame
of the video and using the LSTM network to identify the dance style.
The video is loaded into OpenCV, which processes each frame using the same image
preprocessing techniques that were used to train the model. The processed frame is then passed
through the model to obtain a prediction. The model predicts the dance style by outputting a
class probability distribution across the dance styles it has been trained on. The predicted class
is the class with the highest probability.
