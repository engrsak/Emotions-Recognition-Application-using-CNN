# Emotions Recognition Application using CNN

The **Emotions Recognition Application** is built using a **Convolutional Neural Network (CNN)** to analyze facial expressions and detect a variety of human emotions. This project utilizes deep learning to classify emotions such as happiness, sadness, anger, surprise, and more. It can be applied in areas like customer service, security, and AI-based interactions.

## Features
- Recognizes emotions from real-time facial images.
- Supports emotion classes like happiness, sadness, surprise, anger, etc.
- Built using TensorFlow/Keras with CNN architecture.
- Can be integrated into various applications for human-computer interaction.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/engrsak/emotions-recognition-application-using-cnn.git
    ```

2. Install the required Python packages
3. Download the pre-trained model (or train your own)
4. Run the application:
    ```bash
    python real_time_video.py
    ```


## How it Works

This system uses a **Convolutional Neural Network (CNN)** to process and classify facial expressions into different emotion categories. The following steps summarize the process:

1. **Input Video:** Reading video frames using your camera.
2. **Preprocessing:** The images are preprocessed (resizing, normalization, etc.).
3. **Emotion Prediction:** The CNN model analyzes the image and predicts the emotion.
4. **Output:** The predicted emotion is displayed on the screen.


## Training the Model

If you want to train the model from scratch or improve its accuracy, use the following steps:

1. Prepare a dataset (e.g., the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)).
2. Train the model using the training script:
    ```bash
    python train_emotion_classifier.py
    ```

## Future Improvements

- Add support for more emotions.
- Optimize the model for real-time applications.
- Implement better facial detection and alignment.
- Cross-platform support for different devices.

## License

This project is open-source and licensed under the MIT License.

## Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- TensorFlow/Keras Documentation

## Credits
This work is inspired from [this](https://github.com/oarriaga/face_classification) and was greatly enriched by the insightful resources and tutorials of Adrian Rosebrock.
