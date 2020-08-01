# Captcha Recognition using Recurrent-CNN architecture and CTC Loss

This model is trained on a very small sample of data. It obviously doesn't work for different kinds of captcha images. But the usage of R-CNN along with CTC loss is interesting.
This project was inspired by <a href="https://www.youtube.com/watch?v=IcLEJB2pY2Y">Abhishek Thakur's YouTube video</a>. 

To train the model, run:
<pre>
python3 train.py
</pre>
 
To test the model on an image: 
<pre>
Add the image to the output folder and pass it as an argument.
python3 train.py --filename=test2.png
</pre>
