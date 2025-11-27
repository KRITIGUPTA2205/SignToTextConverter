# SignToTextConverter
🤟 A real-time sign-gesture recognition system that converts hand signs into readable text — helping bridge communication between sign-language users and others.
🎬 Demo Video
🔗 https://github.com/KRITIGUPTA2205/SignToTextConverter/blob/main/DemoSignToText.mp4
Click above to see the demo video of the project in action.
✨ Project Overview

The Sign-to-Text Converter leverages computer vision and deep learning to detect hand gestures (signs) and translate them into human-readable text, in real time.

It supports:

Word-level recognition (dynamic gestures) using LSTM + landmark sequences

Alphabet-level recognition (static gestures/images) using CNN

Ideal for aiding communication for people using sign-based gestures or sign language, especially when a standard interface (text typing) isn’t feasible.
| Component               | Technology / Library                              |
| ----------------------- | ------------------------------------------------- |
| Hand tracking           | MediaPipe Hands                                   |
| Video processing        | OpenCV                                            |
| Word classification     | LSTM (TensorFlow / Keras)                         |
| Alphabet classification | CNN (TensorFlow / Keras)                          |
| Language                | Python                                            |
| Dataset (example)       | Custom word dataset + MNIST Sign Language dataset |
🌱 Future Improvements

Expand the gesture dataset to support a larger vocabulary

Add sentence-level recognition (sequence of gestures → full sentences)

Support two-hand gestures and more complex signs

Add text-to-speech output for accessibility

Create a mobile version (e.g. using TensorFlow Lite + Flutter)

Add support for multiple sign languages
