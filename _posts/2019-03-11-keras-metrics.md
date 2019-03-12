---
layout:     post
title:      Adding Metrics in Keras
date:       2019-03-11 17:00:10
author:     Jacob Richeimer
summary:    Adding metric tensors to a Keras model
categories: keras
thumbnail:  line-chart
tags:
 - keras
 - metric
---

Usually, when training a model, you want to always be able to see what the loss is, so you can have a general idea of how well it's doing. Often times, though, there are other values besides for the loss that you'd also like to keep an eye on throughout the training process. A typical example would be, for a standard classification model, to keep track of the prediction accuracy of the model.

For this purpose, Keras provides the ability to add ["metrics"][1] to a model. The idea is pretty simple. It's the same as a loss in that it's computed at every step within the model's graph, and in that it's displayed to the user throughout the training process, but it in no way affects the actual training of the model like the loss obviously does. Because of the metrics' parallels to the losses, they must be provided to the model just like losses, as functions that take as arguments the outputs of the network and the ground-truth labels and output some (scalar) value.

However, for both losses and for metrics as well, not always can they be easily expressed as functions of outputs and targets. Sometimes it's useful to have the network minimize (partially), or be able to track, some arbitrary tensor that doesn't necessarily depend on the network outputs or on the ground-truth labels. When it comes to losses, Keras somewhat solved this issue by allowing tensors within the graph to be added to layers or models via the `add_loss` function. (Losses that are dependent only on layer weights can also be utilized through Keras's regularizers.) But it has not (yet) provided this ability for metrics.

The following simple function, which I have found useful on several occasions, allows any Keras tensor to be added to the metrics of a particular model.

```python
def add_metric(model, metrics, names=None):
  if not isinstance(metrics, list):
    metrics = [metrics]
  if names != None and not isinstance(names, list):
    names = [names]
  if names is None:
    names = [m.name for m in metrics]
  assert len(names) == len(metrics)
  
  model.metrics_tensors.extend(metrics)
  model.metrics_names.extend(names)
```

Unlike with losses, since the metrics don't affect the rest of the model (like the losses do by adding updates), you don't have to recompile the model after adding metrics if it's already been compiled.

Another way this could be useful is if you have a loss function that's really a combination of two or more objectives. As an example, suppose for whatever reason you wanted your loss to be the weighted sum of binary-cross-entropy and mean-squared-error between the output and the labels. Now, in Keras, that's really one loss function, since it operated on a single pair of output/target. But you would probably want to see each component of the loss independently during training. You could do this as follows:

```python
class MyLoss(weights=[0.5, 0.5]):
  def __init__(self):
    self.weights = weights
  
  def loss_function(self, y_true, y_pred):
    self.bce = keras.losses.binary_crossentropy(y_true, y_pred)
    self.mse = keras.losses.mean_squared_error(y_true, y_pred)
    loss = weights[0] * bce + weights[1] * mse
    return loss
    
  def get_loss_tensors(self):
    return [self.bce, self.mse]
    
my_loss = MyLoss()
model.compile(loss=my_loss.loss_function, optimizer='sgd')
add_metric(model, my_loss.get_loss_tensors(), names=['bce_loss', 'mse_loss'])
model.fit(...)
```

That way, even though you're really only using one loss function for your Keras model, you can keep track of both "internal" losses during training, as both of them (in addition to their combination) will be displayed.

[1]: https://keras.io/metrics/
