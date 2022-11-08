# Crowd Counting 

Crowd counting has been a strong subdomain of computer vision for a long time. There are some datasets such as JHU crowd ++ , ShangaiTech dataset, etc. Using these datasets, some models have been developed to estimate the size of the crowd based on a picture. [This](https://paperswithcode.com/task/crowd-counting) has been one of the hottest field in Computer Vision. As a machine learning engineer at ASU, I was tasked with a similar problem.

## Football Stands

The dataset used to train the models are similar to the pictures shown as follows.

![Picture](https://production-media.paperswithcode.com/datasets/crows.jpg)

The picture from the football stands are as follows

![Picture2](https://www.tempetourism.com/wp-content/uploads/Sun-Devil-Football-Stadium-1024x500.jpg)

There are many differences to the crowd counting in the football stadium and the dataset mentioned above. Some of the difficulties are listed as follows

- The crowd is confined to neatly arranged football stands. The population walking in/out of the stands should not be included.
- There is a presence of a strong background color i.e, color of the seats and that too they are different for different stands
- The stands are not rectangular and hence not "croppable" to a fixed size for model input.
- An entire picture has some text and other colors which are throwing off the model's people count.

Given the above nuances/differences of crowd counting in our application, we have to design the application in the corresponding way to address the problem.

## Solution Architecture

In this section, we will be looking at all the parts - Hardware ans Software to build this project.

### Model
For this crowd counting task , I have used the following [model](https://paperswithcode.com/paper/csrnet-dilated-convolutional-neural-networks) with some extra preprocessing to help the crowd estimation process. The first preprocessing step is to normalize the picture values to their standard normal values.  The first step is to fix the dimensions of the model by using 4K image dimensions. Once the model's nodes and layer sizes are fixed, we can further optimize the model for inference.

### Compute Device

The model is very large but can be optimized to fit into edge computing devices that consume a fraction of the power consumed by desktop GPU. The crowd count does not vary by a large amount in a short period of time. Therefore we can deploy the model on a device that has better speed/power trade off. The device of choice for this project is Google Coral Dev board. 

![Coral](https://lh3.googleusercontent.com/C6HdVBJaaIBhCzzssfkMQJecghiyQ3EJ8b6KW-nCf-VPSA_9HYiZMXG8VJCWwY5P1P0Bl-RtTRd9uNCkmT7ABUnbnL2-IG_izckmTRI=s0)

This device accelerates the inference process after the integer quantization (INT8) of the deep learning model. The quantization can be done by the use of edgetpu_compiler. The Axis camera would be connected to the same network as that of this device, enabling the device to capture the images periodically and send back the people count estimate per stand.

### Software Modeling

The first task was to create software to enable the processing of data. Based on the previous ASU football image, I have create a class called [AudienceStand](https://gist.github.com/gksriharsha/90a2d69f5a1af687eee23b3d4c43770e) to store the particular stand's information in the corresponding object. This includes the coordinates of the stand, fill percentage/count of the stand,etc. 

Similarly, a class is created to store the information and necessary functionalities about the entire stadium. The URL of the camera, numpy object of the image, etc are some of the values. This model of development is very close to the Object Oriented approach of building the solution.

Once the class modeling and sequence of processing is finished , we continue to build the image processing pipeline. There are many ways to process the image to get the crowd % fill in a particular stand. These are implemented in the [gist](https://gist.github.com/gksriharsha/c989af9563ef0df46964129e1643b89b). In this project, deep learning solution is the main focus as other solution is explained in [this](https://gksriharsha.wixsite.com/stadium-project/about-4) webpage.

### Machine Learning Modeling

The most important part of the image processing pipeline is to perform segmentation of each stand from the bigger picture. In this project, opencv's xor function in combination with polylines is used to seperate out the stand. After the stand is seperated, it is stored in the corresponding variable in the AudienceStand class. Based on the heatmap generated it can be seen that the black color produced from the xor operation gives rise to some noise in output summation function. Therefore only the values above the mode of the image is considered for summation. This will increase the accuracy of the model by many fold. In addition, a small light weight RandomForest model is trained to identify the various seats colors present in the stand and reaplce these colors with black color. Since the noise produced from black color is anyways being removed, this would increase the accuracy of the estimate. 

As a fun extension, I have tried to count the number of people supporting Arizona State University (Yellow shirts/hats). In this extension, I have modified the BGR color space to LAB color space where most of the emphasis is present on the yellow color. After a series of experimentaion, a threshold for the ASU's yellow color is determined. All the colors that are failing this threshold are made into black. This will generate a white noise that will be ignore by the last layer's summation.

# Output

I was able to execute this model atleast once per minute using the google coral board. Since this device consumes less than 15 W, it can be made a viable edge computing inference solution for this project. This has been a really thrilling project where I had to work on many performance metrics such as MSE, MAE as well as non-performance metrics such as thermal profile, speed of inference, etc.

The images are as follows:

**Source**

![image](https://lh6.googleusercontent.com/6nln9bZX4W4SNa3mvg5PNYbHCFIGIu-R5GIWQcJ67oVxDTTeB7K7D_IjlpzeOPev4Us=w2400)

**Processed**

![image](https://lh5.googleusercontent.com/bUWqZ6uqQjKyvj6_rc1-tOs0IRWWDOpeA0j_7SgTDAzlFGtUUq01to3YejnIjvkwyXs=w2400)