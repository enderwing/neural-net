# neural-net
A custom neural net implementation I am writing to practice Python. The neural net is a simple, Forward Feeding network that is trained using backpropagation.
The code is currently a little messy, with several vestigal bits that I wrote to use for testing. 
I will clean these up eventually if I continue developing this project passed it's due date.

## Usage
To use the network, run the `main.py` script. By default, there are no trained networks available. One can be created by entering `C` at the first prompt.

For the in class demonstration, I will supply a pre-trained model. To import the pre-trained model, enter `I` at the first prompt and type `recognizeDigits` for the second prompt.

To test a random digit from the test set of MNIST data, enter `T`. To run the network on a user supplied image: 
1. Enter `C`
2. Save the image as a 28x28 pixel grayscale image to the `neural-net/images` directory.
3. Enter the name of the image file, and the network will process the image, classify it, and print the result to console.

(There is no error checking so don't make any typos 👍.)
