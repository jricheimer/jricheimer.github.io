---
layout:     post
title:      Image Resizing Confusion
date:       2019-02-11 07:02:19
author:     Jacob Richeimer
summary:    Many Different Bilinear Resizing Algorithms Cause Confusion
categories: tensorflow
thumbnail:  expand
tags:
 - tensorflow
 - opencv
 - resize
---

Image resizing is one of the most common image operations available. In computer vision applications, it's used all the time. Traditional algorithms call quite often for operating on image pyramids. Convolutional networks which extract global image features are typically restricted to a fixed input size, which means that most of the time, the original image needs to be resized (or sometimes resized and padded in order to maintain aspect ratio) in order to conform. In per-pixel tasks, like segmentation or keypoint detection, often times the output of a network might need to be resized back to the image resolution to be made use of. Or sometimes, resizing operations are incorporated into the network itself as part of a "decoder" module.

When resizing an image, it's necesary to adopt an interpolation strategy, as most target indices will be mapped to subpixel values, and the image intensity at that subpixel needs to be interpolated from the pixels surounding its location. In my experience, bilinear interpolation is the most common when resizing images, especially when enlarging the image. (If the resize is within a convolutional network, nearest neighbor is also common, since there will be further processing done anyway by subsequent convolutional layers.) I have found, though, that many libraries that have implementations of bilinear resizing differ in their standards as to how to implement it, and this has been a source of confusion for myself and many others. So let's take a close look at a few of those relevant to the computer vision community. 

First, let's look at OpenCV, the gold standard for computer vision algorithms. We'll do a simple test in one dimension to try and see what it does. We'll start off with a `1x6` "image" (single channel), with each value equal to its x-index and resize it to double the length to `1x12`.

```python
import cv2
import numpy as np

x = np.array([[0, 1, 2, 3, 4, 5]], dtype='float32')
print cv2.resize(x, (12,1)).tolist()
```
This outputs:
```
[[0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.0]]
```
As we can see, the edges of the resulting "image" keep the same values as the original. The step value between pixels is 0.5, which is to be expected when scaling by two, except for the first and last steps, which are steps of 0.25. What's going on?

Well, OpenCV assumes that when you resize an image, you don't really just want a scaling of the original indices. If that was the case in our example, for instance, the index `4` of the result would map to index `2` of the source image and would have value `2`. Instead, as we saw, it has value `1.75`.  Why doesn't OpenCV want to scale the indices directly? Let's see what happens when we do that with the `warpAffine` function. Using that function, we can apply any affine function to the image indices, and of course that includes a simple scaling:

```python
x = np.array([[0, 1, 2, 3, 4, 5]], dtype='float32')
M = np.array([[2, 0, 0], [0, 1, 0]], dtype='float32')
print cv2.warpAffine(x, M, dsize=(12,1), borderMode=cv2.BORDER_REFLECT).tolist()
```
The result:
```
[[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]]
```
This makes perfect sense. We're starting with index `0`, which maps to `0` in the source image, and then steadily moving up by steps of `1/scale = 0.5`. But this presents a weird artifact at the end (right edge) of the image. Now we have an extra value on the right which copies the edge value. This is because index `10` in the target maps to `5` in the source image, which is the edge of the source image, but then we still have another index to fill, `11`, which maps to `5.5`. So we've moved over the edge of the image and rely on an interpolation border strategy to fill it in; in this case, I chose border reflection. But because of our zero-indexing, we only end up interpolating over the edge on the right side, but not on the left side; so the left side looks normal, whereas on the right side you get that weird artifact. Here's a simple diagram to show what's going on:

![alt text](https://github.com/jricheimer/jricheimer.github.io/raw/master/_data/resize1.png "Direct Index Scaling")

The dots on top represent the pixel values in the source image and the dots on bottom represent where the pixel values will be in the target image. The dotted lines show where in the source image those bottom dots will be interpolated from. As you can see, there's that awkward target pixel on the right side just hanging over the edge.

That's why OpenCV assumes in the `resize` function that you don't want the straight-forward index scaling. Instead, what it does is it considers the value of a pixel in the image to be the value at the "center" of the pixel. You have to think of a pixel as having an "area" of width and length one. That is, if the top left pixel of an image has value `255`, then that value of `255` fills the area between `0` and `1`, and we take the "center" of that pixel, `(0.5, 0.5)` as its real index. Effectively, we're shifting the indices of the image by `0.5`. Here's a corresponding diagram:

![alt text](https://github.com/jricheimer/jricheimer.github.io/raw/master/_data/resize2.png "OpenCV Resize Standard")

Each pixel now inhabits an "area" represented by the squares. The zero point is considered to be on the left-most edge of the first square so that the first pixel value is actually located at `0.5`. OpenCV assumes that you want the left edge of the output image, at `0` to correspond or *align with* the left edge of the source image, at *its* `0`. And the same on the right side; in this case, that means that the right edge of the target image, at `12.0` should align with the right-most side of the source image at `6.0`. But those values are past the "real" edge of the image, as in actual indices, they correspond to `-0.5` and `11.5` in the target image, and `-0.5` and `5.5` in the source image.

So, given an index `i` in the target image, to map it to an index in the source image, we need to shift our index by `0.5`, then scale it, then shift back by `0.5` again. Let's try it in the form of a python `lambda` and see if it matches the OpenCV result.

```python
f = lambda i, s: (i+0.5)*(1./s)-0.5  # i is the target index and s is the scale
print [f(i, 2) for i in range(12)]
```
That prints the following:
```
[-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25]
```
As we can see, now, instead of going over the edge by 0.5 on just the right-hand side, we're going over the edge on both sides equally by `0.25`. OpenCV employs a reflection border mode, so the edges can be changed to the following:
```
[0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5]
```
And in fact, this is exactly what `cv2.resize` returned above.

Most libraries that I've encountered implement one of the above two standards. Either the direct index scaling (which I believe is what `PIL` does) or the OpenCV-style "shift-and scale" approach (which is also followed by `scikit-image`).

Suppose, though, that you want to incorporate bilinear resizing (which is differentiable) in a convolutional network. This is done, for example, in the [DeepLabv3+][1] model for segmentation. (I recommend [this excellent post][2] about visualizing the difference between increasing resolution in convolutional networks via "deconvolution" layers and via resizing followed by standard convolution layers.) This can be done in Tensorflow with the [`tf.image.resize_bilinear`][4] function. Let's see what it does on our toy example from above.

```python
import tensorflow as tf
import numpy as np

x = np.array([[0,1,2,3,4,5]], dtype='float32')
t = tf.constant(x[np.newaxis,:,:,np,newaxis])   # tensorflow needs a batch axis and a channel axis
sess = tf.Session()
print sess.run(tf.image.resize_bilinear(t, [1,12]))[0,:,:,0].tolist()
```
This prints out as follows:
```
[[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]]
```
We've seen this before. This is direct index scaling. It's what `cv2.warpAffine` does when you give it a scaling transform. But it's not what `cv2.resize` does. And, most probably, it's not what you want your network to do when it's resizing feature maps. To avoid this behavior, Tensorflow provides an option called `align_corners` (which defaults to `False`) as an argument to this function. Let's check it out:

```python
x = np.array([[0,1,2,3,4,5]], dtype='float32')
t = tf.constant(x[np.newaxis,:,:,np,newaxis])   # tensorflow needs a batch axis and a channel axis
sess = tf.Session()
print sess.run(tf.image.resize_bilinear(t, [1,12], align_corners=True))[0,:,:,0].tolist()
```
This time, we get:
```
[[0.0, 0.4545454680919647, 0.9090909361839294, 1.3636363744735718, 1.8181818723678589, 2.2727272510528564, 2.7272727489471436, 3.1818182468414307, 3.6363637447357178, 4.090909004211426, 4.545454502105713, 5.0]]
```

It turns out that Tensorflow, like OpenCV, tries to align the left and right edges of the input and output images. But, *unlike* OpenCV, they don't consider the pixel values to represent the "center" of the pixel areas, i.e. they don't shift the index values by a half in their mapping. Here's what it looks like in one of our dot diagrams.

![alt text](https://github.com/jricheimer/jricheimer.github.io/raw/master/_data/resize3.png "Tensorflow Align Corners")

Tensorflow is aligning the target's `0` with the source's `0` on the left, and also the right-most target index, `11`, is being aligned with the right-most source index, which is `5`. So, actually, in this `align_corners` mode, Tensorflow is still scaling the indices without shifting, but they're just changing the expected scale value. To scale from size `6` to size `12` like in our example is not scaling by a factor of `2`, but rather by a factor of `11/5 = 2.2`. You can think of it as the area or "length" of the source image being scaled to the area or length of the target. In out case, the target may have 12 pixel values, or "points", but there are only 11 "units of length" between the points. The same for the source image - there are only 5 "units". So the scale factor becomes 11/5.

This `align_corners` is likely closer to what you want if you're resizing the feature maps in your network. However, there still doesn't seem to be a way to imitate OpenCV's resizing in Tensorflow. 

To take advantage of Tensorflow's `align_corners` mode, a nice approach when enlarging an image is to make it so that `output_size-1` is divisible by `input_size-1`. That way, the scale factor becomes an integer, and this minimizes the amount of output pixels interpolated from subpixels in the source image. To try that, let's just add one to the sizes from our earlier example:
```python
x = np.array([[0,1,2,3,4,5,6]], dtype='float32')
t = tf.constant(x[np.newaxis,:,:,np,newaxis])   # tensorflow needs a batch axis and a channel axis
sess = tf.Session()
print sess.run(tf.image.resize_bilinear(t, [1,13], align_corners=True))[0,:,:,0].tolist()
```
The result is:
```
[[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]]
```
This is a really nice property, and in fact, this is part of [the standard][3] that the DeepLab team uses for their experiments.

I've seen a lot of confusion about this scattered around online and in github issues, and I hope this clears up some of that confusion for anyone caught up in this. Our small example was just one-dimensional for simplicity's sake, but of course this extends to two dimensions trivially; the top and bottom of the image would be treated the same as the left and right.

[For comparison, I quickly tried out MXNet's and PyTorch's bilinear resizing functions. MXNet's `mxnet.ndarray.contrib.BilinearResize2D` is equivalent to Tensorflow's `align_corners` standard. And PyTorch, in it's `torch.nn.Upsample(mode='bilinear')`, also includes an `align_corners` argument, which performs the same as Tensorflow when `align_corners=True`. However, interestingly, when `align_corners=False`, it performs equivalently to OpenCV's `resize` instead of mimicking Tensorflow.]

[1]: https://arxiv.org/abs/1802.02611
[2]: https://distill.pub/2016/deconv-checkerboard/
[3]: https://github.com/tensorflow/tensorflow/issues/6720#issuecomment-298190596
[4]: https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear
