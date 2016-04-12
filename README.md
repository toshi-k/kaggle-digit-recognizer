Kaggle Digit Recognizer
====

My solution in this Kaggle competition ["Digit Recognizer"](https://www.kaggle.com/c/digit-recognizer), 99.5%.


## Model

Deep Convolutional Neural Network

|Layer Type|#Input Channel|#Ouput Channel|FilterSize|Padding|
|:--:|:--:|:--:|:--:|:--:|
|Input| - |1|-|-|
|Convolution(1)|1|64|5|0|
|Leaky ReLU|64|64|-|-|
|Convolution(2)|64|64|5|2|
|Leaky ReLU|64|64|-|-|
|Convolution(3)|64|256|3|1|
|Leaky ReLU|256|256|-|-|
|Convolution(4)|256|1024|3|0|
|Leaky ReLU|1024|1024|-|-|
|Full Connect(1)|1024|256|-|-|
|Leaky ReLU|256|256|-|-|
|Full Connect(2)|256|10|-|-|
|Output|10|-|-|-|
