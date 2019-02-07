---
layout:     post
title:      The Tensorflow Padding Algorithm
date:       2019-02-08 07:02:19
author:     Jacob Richeimer
summary:    Issues with the padding in tf convolutions
categories: tensorflow
thumbnail:  table
tags:
 - tensorflow
 - convolution
 - padding
 - keras
---

Convolutional layers are the main building blocks of most of the models that I work with, so understanding exactly how the operation
works is quite important. Often times - in fact most of the time - in a normal spatial convolution (with filter size greater than one),
zero-padding is applied to the input tensor to the layer so that we can properly control what the output size will be.

In general, if the spatial input size to a conv layer is `(H, W)` (let's ignore the channel dimension here for convenience) and the kernel
size of the convolution is `(k, k)`, then without any padding involved, the output size will be `(H-k+1, W-k+1)`. Thus, if we want a
different output size than that, we have to artificially change `H` and `W` by appending zeros onto the edges of the image. That's called padding.
In a case where a greater-than-one stride is included, then the above output size will be divided by the stride value in each dimension. 

What's interesting is that in the Tensorflow framework (and Keras as well), instead of
the user specifying how much to pad the image in each dimension, in order for the user to not have to figure that out on their
own, it takes just a padding "mode" as an argument to it's convolution operators. The user can choose between the "same" mode or
the "valid" mode for padding; these specify what the user wants the output size of the convolutional layer to be. If you want the
output size to be the same as the input size (scaled by the stride), then choose "same". If you don't care about the output size, or you care more that only real data be 
considered by the network (rather than "fake" added zeros), then no padding will be done and Tensorflow will compute the "natural"
convolution.

This has consequences, as the exact number of pixels which Tensorflow is padding is hidden under the hood, and as it turns out,
in some cases, their choice is different than what would occur typically in other deep learning frameworks.

Suppose I have a convolutional network where I expect the input size to be even-numbered.
(This could be for either height or width, but for simplicity, let's say these images are square.)
As an example, let's say the image size is `(256, 256)`. The first convolutional layer in the network is usually strided so as to
reduce the resolution as quickly as possible. Let's say we use a stride of two. It's common in a case like this to use the "same"
padding mode in Tensorflow so we can control the output to be of size `(128,128)`, as this might be important
for the dimensionality of the features at the end of the network. And let's say the filter size is `(3, 3)`.
What does Tensorflow do in this case?

As the [documentation][1] specifies, in a case like this where the input size is divisible by the stride (`128 mod 2 = 0`),
only a padding of `k - s` is necessary, where `k` is the filter size and `s` is the stride. Here, that means that a padding of
`3 - 2 = 1` is needed. That means the image will be padded to be size `(257, 257)`. Elsewhere in the docs, it notes that when the
padding number is odd, like it is in our case, more zeros will be added to the end (i.e. the bottom and the right) than to the beginning
(i.e. the top and the left). So in our case, an extra row of zeros will be padded to the bottom of the image, and an extra column
will be padded on the right, but nothing will be padded on the top or on the left. The padding is asymmetrical.

This issue can cause confusion when converting networks into Tensorflow which were trained in other frameworks.
Or even loading a Keras model which was trained using a different Keras backend.
In most other libraries, like PyTorch or MXNet, the user has to actually specify exactly how much to pad the input, and it's very uncommon for people
to pad asymetrically. I suppose it's just not the way we think, even though it's technically more efficient to do so. (Note: Theano
actually allows both pre-defined padding "modes" or specific padding numbers as arguments for their convolution layers. However, their
version of the "same" mode is called the "half" mode, which only employs symmetrical padding, unlike Tensorflow.)

If you have an pre-trained MXNet model which has a convolutional layer like the one in our example, which operates on an even-sized input with an odd-sized
kernel and an even stride, you'll find most of the time the padding will be symmetrical. So if you want to convert it into a Keras
model to be used with a Tensorflow backend, instead of using just a `Conv2D` layer with `padding='same'`, use first a `ZeroPadding2D(padding=(1,1))`
layer to imitate the padding done in the MXNet model, then apply the `Conv2D` layer with `padding='valid'`.

In fact, if you use Keras for your models, and expect to be switching back and forth between backends (Keras supports Tensorflow, Theano, and CNTK backends; and there's a [fork of Keras][2]
which includes a MXNet backend as well), then the safe way to apply convolutional operators is to use this method of first applying
a `ZeroPadding2D` layer, then the `Conv2D`. That way there's no ambiguity in the padding numbers, and you can expect the results of
inference to be the same even when switching between backends.





[1]: https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
[2]: https://github.com/awslabs/keras-apache-mxnet
