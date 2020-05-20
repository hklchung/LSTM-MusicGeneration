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
* [Result](#result)
* [Contributing](#contributing)
* [Contact](#contact)
* [Upcoming Features](#upcoming-features)
* [Known Issues](#known-issues)

<!-- ABOUT THE PROJECT -->
## About the Project
In this project, I aim to build up a sequence model using LSTM to automatically produce musical compositions. The model is currently capable of generating music by predicting the most likely next note when given a sequence of notes, based on music compositions it has been trained on. Separate to this, the model is also capable of predicting the duration of each note in a similar manner.

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

<!-- RESULT -->
## Result
Here are some sample clips from trained models.
* LSTM model predicts pitch only
[Sample output](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output3.mid)
* LSTM model predicts pitch + note duration
[Sample output](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output4.mid)
* LSTM model predicts pitch + note duration + offset between notes
[Sample output A](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output6.mid)
[Sample output B](https://github.com/hklchung/LSTM-MusicGeneration/blob/master/Result/output8.mid)

<!-- CONTRIBUTING -->
## Contributing
I welcome anyone to contribute to this project so if you are interested, feel free to add your code.
Alternatively, if you are not a programmer but would still like to contribute to this project, please click on the request feature button at the top of the page and provide your valuable feedback.

<!-- CONTACT -->
## Contact
* [Leslie Chung](https://github.com/hklchung)

<!-- UPCOMING FEATURES -->
## Upcoming features
* Predict loudness/softness of notes
* ~~Adding silence between notes~~ (part of updates on 20/05/2020)

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
