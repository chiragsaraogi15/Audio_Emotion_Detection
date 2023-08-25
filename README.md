# Audio Emotion Detection

Audio Emotion Detection is a project aimed at classifying audio tracks into different emotion categories using machine learning techniques. The project uses the GTZAN dataset for music genre classification.

## Dataset

The project utilizes the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) from Kaggle. This dataset contains audio tracks categorized into various music genres.

## Project Overview

The project workflow is organized into several steps:

1. **Emotion Clustering**:
   In the initial step, the dataset is subjected to KMeans clustering to identify underlying patterns and group similar audio tracks based on their features. The results of this step help in understanding the intrinsic structure of the data.

2. **Emotion Classification**:
   Based on the clustering results, an XGBoost classification model is constructed. This model is trained to classify audio tracks into their respective emotion categories. This step leverages the insights gained from the clustering process.

3. **Feature Extraction**:
   The project also includes code to extract relevant audio features using the `librosa` library. These features are then fed into the trained classification model to predict the emotion associated with each audio track.

## Notebooks

The project is organized into the following Jupyter Notebook files:

1. **1_Emotion_Clustering.ipynb**:
   This notebook contains the code for performing KMeans clustering on the GTZAN dataset. The objective is to group similar audio tracks based on their features and identify potential emotion clusters.

2. **2_Emotion_Classification.ipynb**:
   Here, an XGBoost classification model is built using the clustered data. This model is trained to classify audio tracks into different emotion categories based on their features.

3. **Audio_Emotion_Detection.ipynb**:
   This notebook demonstrates the complete process of audio emotion detection. It includes code to extract audio features using the `librosa` library, and then applies the trained classification model to predict the emotion associated with the audio tracks.

Feel free to explore these notebooks for a detailed understanding of each step.

## Usage

To replicate the project and explore the notebooks, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/chiragsaraogi15/Audio_Emotion_Detection.git
   cd Audio_Emotion_Detection
