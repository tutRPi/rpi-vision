# RPI Vision

Deep object detection on a Raspberry Pi using Tensorflow & Keras.

### Materials

* Raspberry Pi 4
* SD card 8+ GB
* 3.5" 480 x 320 PiTFT display (https://www.adafruit.com/product/2441)

### Install Dependencies (on Raspberry Pi)

Follow the guide in our Learn System
https://learn.adafruit.com/running-tensorflow-lite-on-the-raspberry-pi-4

### Running a trainer (GPU Accelerated)

```
pip install -r trainer.requirements.txt
```

### Analyzing via Tensorboard

```
tensorboard --logdir gs://my-gcs-bucket/my-model/logs/
```

### References

* [Training a neural network from little data](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
* [How to easily Detect Objects with Neural Nets](https://medium.com/nanonets/how-to-easily-detect-objects-with-deep-learning-on-raspberrypi-225f29635c74)
* [d4, d6, d8. d10, d20 images](https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images)
