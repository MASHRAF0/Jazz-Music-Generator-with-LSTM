# Jazz-Music-Generator-with-LSTM

In this project, we will implement a model that uses an LSTM to generate music. we will even be able to listen to our own music at the end of the assignment.

<img src="images/jazz.jpg" style="width:450;height:300px;">

We will Apply that by : 

1.Apply an LSTM to music generation. 2.Generate your own jazz music with deep learning.

Please run the following cell to load all the packages required in this assignment. This may take a few minutes. (refer lstm.py) 
1 - Problem statement We would like to create a jazz music piece specially for a friend's birthday. However, we don't know any instruments or music composition. Fortunately, we know deep learning and will solve this problem using an LSTM netwok.

We will train a network to generate novel jazz solos in a style representative of a body of performed work.

1.1 - Dataset We will train your algorithm on a corpus of Jazz music. Run the cell below to listen to a snippet of the audio from the training set: (refer lstm.py)

We have taken care of the preprocessing of the musical data to render it in terms of musical "values." you can informally think of each "value" as a note, which comprises a pitch and a duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this--specifically, it also captures the information needed to play multiple notes at the same time. For example, when playing a music piece, you might press down two piano keys at the same time (playng multiple notes at the same time generates what's called a "chord"). But we don't need to worry about the details of music theory for this assignment. For the purpose of this assignment, all you need to know is that we will obtain a dataset of values, and will learn an RNN model to generate sequences of values.

Our music generation system will use 78 unique values. Run the following code to load the raw music data and preprocess it into values. This might take a few minutes.(refer lstm.py)

Expected output: shape of X: (60, 30, 78) number of training examples: 60 Tx (length of sequence): 30 total # of unique values: 78 Shape of Y: (30, 60, 78)

You have just loaded the following:

X: This is an (m, Tx , 78) dimensional array. We have m training examples, each of which is a snippet of Tx=30 musical values. At each time step, the input is one of 78 different possible values, represented as a one-hot vector. Thus for example, X[i,t,:] is a one-hot vector representating the value of the i-th example at time t.

Y: This is essentially the same as X, but shifted one step to the left (to the past). Similar to the dinosaurus assignment, we're interested in the network using the previous values to predict the next value, so our sequence model will try to predict y⟨t⟩ given x⟨1⟩,…,x⟨t⟩ . However, the data in Y is reordered to be dimension (Ty,m,78) , where Ty=Tx . This format makes it more convenient to feed to the LSTM later.

n_values: The number of unique values in this dataset. This should be 78.

indices_values: python dictionary mapping from 0-77 to musical values.

1.2 - Overview of our model Here is the architecture of the model we will use. This is similar to the Dinosaurus model you had used in the previous notebook, except that in you will be implementing it in Keras. The architecture is as follows: (see images)

We will be training the model on random snippets of 30 values taken from a much longer piece of music. Thus, we won't bother to set the first input x⟨1⟩=0→ , which we had done previously to denote the start of a dinosaur name, since now most of these snippets of audio start somewhere in the middle of a piece of music. We are setting each of the snippts to have the same length Tx=30 to make vectorization easier.

2 - Building the model In this part you will build and train a model that will learn musical patterns. To do so, you will need to build a model that takes in X of shape (m,Tx,78) and Y of shape (Ty,m,78) . We will use an LSTM with 64 dimensional hidden states. Lets set n_a = 64.(refer lstm.py)

Here's how you can create a Keras model with multiple inputs and outputs. If you're building an RNN where even at test time entire input sequence x⟨1⟩,x⟨2⟩,…,x⟨Tx⟩ were given in advance, for example if the inputs were words and the output was a label, then Keras has simple built-in functions to build the model. However, for sequence generation, at test time we don't know all the values of x⟨t⟩ in advance; instead we generate them one at a time using x⟨t⟩=y⟨t−1⟩ . So the code will be a bit more complicated, and you'll need to implement your own for-loop to iterate over the different time steps.

The function djmodel() will call the LSTM layer Tx times using a for-loop, and it is important that all Tx copies have the same weights. I.e., it should not re-initiaiize the weights every time---the Tx steps should have shared weights. The key steps for implementing layers with shareable weights in Keras are:

Define the layer objects (we will use global variables for this). Call these objects when propagating the input. We have defined the layers objects you need as global variables. Please run the next cell to create them. Please check the Keras documentation to make sure you understand what these layers are: Reshape(), LSTM(), Dense(). (refer lstm.py)

Each of reshapor, LSTM_cell and densor are now layer objects, and you can use them to implement djmodel(). In order to propagate a Keras tensor object X through one of these layers, use layer_object(X) (or layer_object([X,Y]) if it requires multiple inputs.). For example, reshapor(X) will propagate X through the Reshape((1,78)) layer defined above.
