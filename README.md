[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![Keras 2.3.1](https://img.shields.io/badge/keras-2.3.1-green.svg?style=plastic)
![Music21 5.7.2](https://img.shields.io/badge/music21-5.7.2-green.svg?style=plastic)
![Pygame 1.9.6](https://img.shields.io/badge/pygame-1.9.6-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

<br />
<p align="center">
  <a href="https://github.com/hklchung/LSTM-MusicGeneration">
    <img src="https://i.pinimg.com/originals/c9/6d/b2/c96db2b4d8fe3ae4a962c225b40c30a2.jpg" height="200">
  </a>

  <h3 align="center">Music Generation</h3>

  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [The Code](#the-code)
  * [Training Tips](#training-tips)
* [Result](#result)
* [Contributing](#contributing)
* [Contact](#contact)
* [Upcoming Features](#upcoming-features)
* [Known Issues](#known-issues)

<!-- ABOUT THE PROJECT -->
## About the Project
In this project, I aim to build up a sequence model using LSTM to automatically produce musical compositions. The model is currently capable of generating music by predicting the most likely next note when given a sequence of notes, based on music compositions it has been trained on. Separate to this, the model is also capable of predicting the duration of each note and the time difference between notes in a similar manner.

<!-- GETTING STARTED -->
## Getting Started
To get started, please ensure you have the below packages installed.

<!-- PREREQUISITES -->
### Prerequisites
* Keras==2.3.1
* Numpy==1.18.2
* Pandas==1.0.3
* Music21==5.7.2
* Pygame==1.9.6

<!-- THE CODE -->
### The Code
Here is a summary of what I did in the code [main.py](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/main.py).
1. Load in the required packages (see above listed prerequisites)
2. Define music player functions to play .mid files from folder and in session midi objects using pygame package
3. Define a function to visualise model loss over epoch
4. Load .mid tracks from /Music folder and breakdown data in the following lists
  a) <b>notes</b>
  b) <b>duration</b>
  c) <b>delta</b> (offsets)
5. One hot encode our data, e.g. say we have a total of 40 possible notes in our loaded track with 100 notes (track length) then our one hot encoded array will have shape (40, 100). Additionally, we will breakdown our track into segments each containing 50 notes, i.e. first segment will have notes from position 0 to 49, second segment will have notes from position 1 to 50 and so on. Therefore, in this example our final notes array will have shape (60, 50, 40) for 60 segments, each 50 notes long, each note having 40 possibilities.
6. We do the same for duration and delta, then stack notes, duration and delta arrays into one
7. Define our LSTM model architecture with one input and 3 outputs, corresponding to our notes, duration and delta (offset) predictions
8. Train the model
9. Use the model to "create" a new song of length 100 notes
  a) randomly pick a sequence from the training data
  b) use said sequence to predict the next note, duration and delta
  c) remove first note from the sequence and insert the predicted info to the end of the sequence
  d) use this new sequence to predict the next note, duration and delta
  e) repeat until we have 100 predicted notes
  f) translate the predicted notes into midi format and save file
  
<!-- TRAINING TIPS -->
### Training Tips
* Training a model with more songs tend to produce worse results, this is because your songs are probably not written in the same key or tempo
* For training, pick songs that are not too fast or slow, and preferably with clear melody, as opposed to blues, jazz, bossa nova, etc

<!-- RESULT -->
## Result
Here are some sample clips from trained models. Try to listen to the clips in order so that you can fully appreciate the benefits of adding extra predictive capabilities (note duration + note offset) in the LSTM model.
* LSTM model predicts pitch only <br>
[Sample output](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output3.mid)
* LSTM model predicts pitch + note duration <br>
[Sample output](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output4.mid)
* LSTM model predicts pitch + note duration + offset between notes <br>
[Sample output A](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output6.mid) <br>
[Sample output B](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output8.mid) <br>
[Sample output B (YouTube link)](https://youtu.be/2mIWyfcEPcQ)

<!-- CONTRIBUTING -->
## Contributing
I welcome anyone to contribute to this project so if you are interested, feel free to add your code.
Alternatively, if you are not a programmer but would still like to contribute to this project, please click on the request feature button at the top of the page and provide your valuable feedback.

<!-- CONTACT -->
## Contact
* [Leslie Chung](https://github.com/hklchung)

<!-- UPCOMING FEATURES -->
## Upcoming features
* ~~Adding silence between notes~~ (part of updates on 20/05/2020)
* Predict loudness/softness of notes
* Predict complementing notes from other instruments, e.g. violin

<!-- KNOWN ISSUES -->
## Known issues
* Code is currently a bit of a mess as I hacked through most of it
* Quality of the predicted notes seem to have suffered after addition of note duration and note offset prediction capabilities

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/hklchung/LSTM-MusicGeneration.svg?style=flat-square
[contributors-url]: https://github.com/hklchung/LSTM-MusicGeneration/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hklchung/LSTM-MusicGeneration.svg?style=flat-square
[forks-url]: https://github.com/hklchung/LSTM-MusicGeneration/network/members
[stars-shield]: https://img.shields.io/github/stars/hklchung/LSTM-MusicGeneration.svg?style=flat-square
[stars-url]: https://github.com/hklchung/LSTM-MusicGeneration/stargazers
[issues-shield]: https://img.shields.io/github/issues/hklchung/LSTM-MusicGeneration.svg?style=flat-square
[issues-url]: https://github.com/hklchung/LSTM-MusicGeneration/issues
