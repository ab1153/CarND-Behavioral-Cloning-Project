
** Behavrioal Cloning Project **


### Files in project

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.json** and **model.h5** containing a trained convolution neural network model and weights
* **writeup_report.md** summarizing the results
* **track1.mp4** and **track2.mp4**, the video files showing the performance on the both tracks in the simulator

** How to train the model and test the drive **

To train the model:
> python model.py

To run the model and test the drive:
> python drive.py model.json

The model.py loads the data preproeessed from the *xs.npy* and *ys.npy* files.

### Model Architecture and Training Strategy 

- My model consists of four convolutional layers followed four fullly connected layers, where the ELU is used for activation function. 
- The data is normalized in the model using a Keras lambda layer. (model.py line 41-42)
- The model contains dropout layers in two FC-layers. (model.py line 59, 63)
- I have initially split the dataset in training and validation. During the training phase I found the model is hard to overfit because I saw the loss on the validation dataset is alwasy smaller than the loss on the traning dataset. Since the dataset I use is relatively small and we evaluate the performance by running the mode in the simulator, I use all the data to train the model at last.
- The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74).
- I used the dataset provided by the Udacity along with some laps driving data myself including recovery data.


### Architecture and Training Documentation 

** Solution Design Approach **

The objective is to approach a regression function which is able to output appropriate steering angles to keep the car drive between the lanes. The network should output the continous value, so the mean squared error is used as objective function.

I began with the raw dataset from the Udacity. The models with various convolutional layers and fully connected layers were used. After some failing experiments I thounght since the dataset is extremly biased toward the zero angle the key is to balance the dataset for training. And in such a simple road situation the complexity of model might not play a big part.

I collected some data by myself as well as some recovery data. These along with the data from Udacity comprise my final training dataset.

I saw the people and in the nvidia paper where they used both side camera and added a correction to the corresponding angles. I can't calculate the amount added to the steering angles. Some of these corrections range from 0.1 to 0.25. These to me are just magic numbers. I decided to just use the center camera.

In the nvidia paper it was mentioned the FC-layers might play a part in classifying images instead of just controller. So in my final simple model I assign more parameters on the FC-layers. With the dropout on these FC-layers my model generalized well on the second track.

How the dataset is selected and balanced see the section **Creation of the Training Set & Training Process**.

** Final Model Architecture **

My model looks as following: 


```python
import model
m = model.pred_steering()
m.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    lambda_1 (Lambda)                (None, 20, 80, 3)     0           lambda_input_1[0][0]             
    ____________________________________________________________________________________________________
    convolution2d_1 (Convolution2D)  (None, 16, 78, 8)     368         lambda_1[0][0]                   
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 16, 78, 8)     0           convolution2d_1[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 12, 76, 12)    1452        activation_1[0][0]               
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 12, 76, 12)    0           convolution2d_2[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_1 (MaxPooling2D)    (None, 12, 38, 12)    0           activation_2[0][0]               
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 10, 36, 12)    1308        maxpooling2d_1[0][0]             
    ____________________________________________________________________________________________________
    activation_3 (Activation)        (None, 10, 36, 12)    0           convolution2d_3[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 8, 34, 12)     1308        activation_3[0][0]               
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 8, 34, 12)     0           convolution2d_4[0][0]            
    ____________________________________________________________________________________________________
    maxpooling2d_2 (MaxPooling2D)    (None, 4, 17, 12)     0           activation_4[0][0]               
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 816)           0           maxpooling2d_2[0][0]             
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 512)           418304      flatten_1[0][0]                  
    ____________________________________________________________________________________________________
    activation_5 (Activation)        (None, 512)           0           dense_1[0][0]                    
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 512)           0           activation_5[0][0]               
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 128)           65664       dropout_1[0][0]                  
    ____________________________________________________________________________________________________
    activation_6 (Activation)        (None, 128)           0           dense_2[0][0]                    
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 128)           0           activation_6[0][0]               
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 64)            8256        dropout_2[0][0]                  
    ____________________________________________________________________________________________________
    activation_7 (Activation)        (None, 64)            0           dense_3[0][0]                    
    ____________________________________________________________________________________________________
    dense_4 (Dense)                  (None, 1)             65          activation_7[0][0]               
    ====================================================================================================
    Total params: 496725
    ____________________________________________________________________________________________________
    

** Creation of the Training Set & Training Process **

The file **data.py.ipynb** includes the code which shows how the data are prepared.

The data I collected include 42282 frames. Since I used only center camera only 14094 of them were the candidates.
The 14094 frames are split uniformly in 100 bins on the corresponding steering angles. For each bin maximal 100 frames were sampled. The bins without steering angle are discared. If the samples in some bin were not enough extra data are created to satisfying the minimal size (50) for that bin. The extra data were created by randomly using smoothing filter, brightness adjustments and applying random patches of adjusted brightness.

At the end 5164 samples of the 14094 were saved in npy files. In the file **model.py** what the generator did is just to flip the every batch frame and output the corresponding inversed steering angle (line 33-34). In effect 10304 samples were used to train the model.



