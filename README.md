# Predict xorshift128
```xorshift128``` is a subclass of LFSRs. ```128``` bits is used for the state representation and is used to generate the next state in a pseudorandom fashion. Simple neural networks can be used to predict subsequent states by only using the current state, without knowing the seed.

The implementation is based of an [article](https://research.nccgroup.com/2021/10/15/cracking-random-number-generators-using-machine-learning-part-1-xorshift128/) written by NCC Group.


## Usage
Run ```python main.py``` and you should see some output like this:
```
Generating RNG samples: 100%|██████| 2000000/2000000 [00:32<00:00, 60713.67it/s]
Training model (epoch 100): 100%|█| 100/100 [1:15:08<00:00, 45.09s/it, 97.33% bi
Testing trained model with new random seed: [ 888456226   16328602 4161758075 4048941973]
Correctly predicted 25/100 samples
```
Training the model takes about an hour on my machine without GPU acceleration. It does not reach 100% accuracy like the article does, but it is still able to predict correct 1/4 of the times. The discrepancy is likely caused by using a different optimizer (```Adam``` instead of ```NAdam```), and there is no hyperparameter tuning for the optimizer.

A potential improvement would be to use a quantized neural network which are better suited for bit-wise operations.