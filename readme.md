# Project Name

This is a program that will detect stairs that is in the frame of the camera and highlight them using the DetectNet Object Detection model for those with vision impairments. The real-world application of this would be implementing this application in knee brace so that when there is a step or stairs detected, it lets them know with a voice system that I will be able to incorporate in my program soon. This would prevent falls for people with visual impairments as well as elders in general, as [research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4636376/) found that stairs have been rated as one of the top five places that provide difficulties and an [average of 12,000 deaths per year](https://sobolaw.com/common-injuries-from-falling-down-stairs/#:~:text=According%20to%20this%20study%2C%20falls,of%2012%2C000%20deaths%20per%20year). Currently, the program is still a work in progress, but the ultimate goal is to connect it to a generative text model that generates AI commentary on an earset connected to a knee brace. In addition, I’d like to train the model to identify more objects that can cause accidents for visually impaired people, such as sharp objects, danger signs, broken glass, manholes, and fires with greater accuracy.


## The Algorithm

This program uses DetectNet, a convolutional neural network that excels at identifying patterns in images and detecting objects. The network is comprised of various convolutional layers, each containing multiple filters that classify individual pixels. Together, these layers can divide an image into different sections based on distinct patterns, allowing the program to identify and locate specific objects. The more layers the network has, the more complex and specialized the objects it can detect. I created my own DNN model by performing transfer learning on an existing model called ssd-mobilenet. Retraining this model invovles fine-tuning the parameters of last layers to recognize new class(es) of object(s), which in my case is stairs. I followed the instructions inside [jetson-inference library](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md) that gives you a step-by-step instructions on how to retrain the preexiting model.
1. I worked inside jetson-inference library docker container that includes all other sub-libraries and modules required to perform retraining.
2. I downloaded a dataset of stairs from [open images dataset](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0fp6w) and performed transfer learning with train_ssd.py script sitting inside the jetson-inference library.
3. I converted the model to onnx, a format that can be loaded into my own python program.
4. Inside my python program, I changed the default network to my onnx model from the default ssd-mobilenet model and changed the elements in the parameters such as where to point for its labels file. My program automatically gets the input from webcam, which sits as "/dev/video0" but it can be altered if you choose to do so. Then it outputs into a mp4 file. 

## Running this project

You have the option of connecting through your terminal (Mac), Putty (Windows), or through your favorite IDE but the instructions here will be on running through IDE

1. Connect to your Jetson Nano using by SSH in your favorite IDE (in my case, it was VSCode)

2. Clone the "StairDetection" project into your home folder:

```bash
git clone https://github.com/dusty-nv/jetson-inference
```

[View a video explanation here](video link)
