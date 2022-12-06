# neural-net
A custom neural net implementation I am writing to practice Python. The network could be trained with any data,
I am training it with handwriting data of the 10 digits so that it will recognize other handwritten digits.

The neural net is a simple, Forward Feeding network that is trained using backpropagation.
The code is currently a little messy, with several vestigial bits that I wrote to use for testing. 
I will clean these up eventually if I continue developing this project passed its due date.

## Usage
Note: This project requires the `python-mnist` package to import the MNIST data. Ensure **not** to install the package simply called `mnist` on accident. Other common packages are also required, see the bottom of the README.

To use the network, run the `main.py` script. By default, there are no trained networks available. One can be created by entering `C` at the first prompt.

For the in class demonstration, I will supply a pre-trained model. To import the pre-trained model, enter `I` at the first prompt and type `recognizeDigits` for the second prompt.

To test a random digit from the test set of MNIST data, enter `T`. To run the network on a user supplied image: 
1. Enter `C`
2. Save the image as a 28x28 pixel grayscale image to the `neural-net/images` directory.
3. Enter the name of the image file (*without* file extension), and the network will process the image, classify it, and print the result to console.

(There is no error checking so don't make any typos 👍.)

## Examples
The network classifies images that are 28x28 pixels, grayscale, and use white (255) as the foreground and black (0) as the background. It should also have the digit centered in a certain way for the network to work optimally. For example:

<img width="150" height="150" src="https://user-images.githubusercontent.com/5438811/205838240-338c3162-96b6-486e-bb39-86ad879929f7.png">

Here is an image that is prepared correctly for the network to process. (The network trained for class today correctly classified this as an `8`.)

There is pre-processing code to invert images that are black text on a white background, and to center the digit in the image. Below is the pre-pre-processed image from above:

<img width="150" height="150" src="https://user-images.githubusercontent.com/5438811/205839311-b55981e9-1e4b-4e78-b6b2-1746cced8673.png">

---
### Credits
- Training and testing image data: The [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- General information and project structure: [This wonderful video](https://www.youtube.com/watch?v=hfMk-kjRv4c) by Sebastian Lague
- Specific information about various activation functions and network optimization: [Machinelearningmastery.com](https://machinelearningmastery.com/)
---
### All required external packages
(Starred packages indicate packages which are less likely to be automatically installed)
- `copy`*
- `math`
- `numpy`
- `opencv-python`*
- `os`
- `platform`
- `python-mnist`*
- `random`
