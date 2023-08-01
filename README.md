# Audio-Signal-Processing-ML

librosa/Audio Signal Processing and Machine Learning for Sound Classification using Mel-frequency cepstral coefficients (MFCCs) and Random Forest Classifier.

1. Importing the required libraries:
   - `librosa` and `librosa.display`: For audio processing and visualization.
   - `matplotlib.pyplot` and `matplotlib.patches.Rectangle`: For plotting and drawing rectangles.
   - `numpy`: For numerical computations.
   - `pandas`: For data manipulation and analysis.
   - `sklearn` modules: For machine learning tasks.

2. Loading and Processing the Audio:
   - The code loads an audio file named 'Baph.wav' using `librosa.load` and stores the audio data in the `audio` variable and the sampling rate in `sr`.
   - It calculates the spectrogram using the Short-Time Fourier Transform (STFT) and stores it in the `spectrogram` variable.
   - It then calculates the Mel-frequency cepstral coefficients (MFCCs) using `librosa.feature.mfcc`.

3. Plotting the Spectrogram and MFCCs:
   - The code uses `librosa.display.specshow` to plot the spectrogram in dB scale.
   - It plots the MFCCs using `librosa.display.specshow`.

4. Saving MFCCs as a CSV file:
   - The MFCCs are saved as a CSV file named 'Baph_mfccs_data.csv'.

5. Reading MFCCs from the CSV file:
   - The code reads the saved CSV file back into a DataFrame named `df_mfccs`.

6. Data Analysis and Exploration:
   - The code calculates descriptive statistics of the MFCC data using `df_mfccs.describe()`.
   - It then plots the MFCC coefficients over time.

7. Machine Learning Model (Random Forest Classifier):
   - The code uses the MFCC data from the CSV file to build a Random Forest Classifier.
   - The class labels are converted to integers using `LabelEncoder`.
   - The data is split into training and testing sets.
   - A Random Forest Classifier with 100 estimators is created and trained on the training data.
   - Predictions are made on the test data, and the accuracy of the model is calculated.

8. Visualization in 3D:
   - The first three MFCC coefficients are used to visualize the data in a 3D scatter plot.
   - The `mpl_toolkits.mplot3d` library is used for 3D plotting.
   - The class labels are represented by colors in the plot.
