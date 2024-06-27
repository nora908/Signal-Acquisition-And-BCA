import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import QColor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import serial
import csv

arduino = serial.Serial('COM9', 9600)



class EEGClassifier:
    def __init__(self, filename='eeg_readings.csv'):
        self.df = pd.read_csv(filename)
        self.features = self.df.drop('y', axis=1)
        self.target = self.df['y']
        self.scaler = StandardScaler()
        self.scaled_features = None
        self._preprocess_data()  # Call preprocessing function during initialization
        self.consecutive_ones_count = 0



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Set fixed geometry for graphicsView
        self.graphicsView = pg.PlotWidget(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")

        # Set fixed geometry for graphicsView_2
        self.graphicsView_2 = pg.PlotWidget(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")

        # Create a layout for the central widget
        central_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        central_layout.setContentsMargins(0, 40, 0, 0)  # Set top margin to 80

        # Set size policy and stretch factors for the widgets
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.graphicsView.setSizePolicy(size_policy)
        self.graphicsView_2.setSizePolicy(size_policy)

        # Add widgets to the layout with stretch factors
        central_layout.addWidget(self.graphicsView, 1)
        central_layout.addWidget(self.graphicsView_2, 1)

        # Set the central layout for the central widget
        self.centralwidget.setLayout(central_layout)
        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.widget1 = self.graphicsView
        self.widget2 = self.graphicsView_2

    def plot_all_channels_data(self):
        # Get data from all electrodes (excluding the 'y' column)
        electrode_data = self.df.drop('y', axis=1)
        self.widget1.clear()

        # Base color (you can adjust this to get different hues)
        base_color = QColor(100, 255, 255)  # You can adjust the hue value (100) to get different hues

        # Plot data from all electrodes using distinct colors
        for idx, column in enumerate(electrode_data.columns):
            channel_data = electrode_data[column]
            hue_shifted_color = base_color.darker(idx * 10)  # Adjust 10 to change color variation
            color = pg.mkColor(hue_shifted_color.red(), hue_shifted_color.green(), hue_shifted_color.blue())
            self.widget1.plot(channel_data.index, channel_data.values, pen=pg.mkPen(color=color))

    def plot_filtered_signals(self):
        # Clear the second plot widget before plotting new data
        self.widget2.clear()

        # Get the number of channels (columns) in the filtered data
        num_channels = self.scaled_features.shape[1]

        # Define base color for plotting signals
        base_color = QColor(255, 100, 100)  # Set color for the signals

        # Plot filtered signals for all channels
        for channel_idx in range(num_channels-1):
            channel_data = self.scaled_features[:, channel_idx]
            hue_shifted_color = base_color.darker(channel_idx * 10)  # Adjust 10 to change color variation
            color = pg.mkColor(hue_shifted_color.red(), hue_shifted_color.green(), hue_shifted_color.blue())
            self.widget2.plot(channel_data, pen=pg.mkPen(color=color))



    def signal_preprocessing(self, df):

        #Removing the outliars
        columns = list(df)
        for col in columns[0:-1]:


            q1 = df[col].quantile(0.10)
            q8 = df[col].quantile(0.80)

            df[col] = np.where(df[col] > q8, q8, df[col])
            df[col] = np.where(df[col] < q1, q1, df[col])

        # Define filter parameters
        lowcut = 1  # Lower cutoff frequency in Hz
        highcut = 50.0  # Upper cutoff frequency in Hz
        nyquist = 0.5 * 256  # Nyquist frequency (half of the sampling rate, assuming 256 Hz)
        low = lowcut / nyquist #Normalizing the lower cutoff frequency.
        high = highcut / nyquist

        # Apply bandpass filter to remove unwanted frequencies
        filtered_data = []
        for column in df.columns:
            channel_data = df[column].values
            b, a = butter(4, [low, high], btype='band')
            filtered_channel_data = filtfilt(b, a, channel_data)
            filtered_data.append(filtered_channel_data)

        filtered_data = np.array(filtered_data).T  # Transpose the array back to (samples, channels) shape

        # Placeholder for artifact removal (example: removing data above a threshold)
        threshold = 100
        artifact_removed_data = np.clip(filtered_data, -threshold, threshold)  # Clip values above/below the threshold

        #In summary, the first part filters the EEG data to retain specific frequency components,
        # while the second part removes artifacts by limiting the amplitude of the filtered data points.
        # ]These steps are common preprocessing techniques used to prepare EEG data for further analysis or modeling.

        preprocessed_data =  artifact_removed_data
        return preprocessed_data



    def calculate_power_spectral_density(self, epoch):
        # Perform FFT to calculate PSD
        n_samples = len(epoch)
        fft_values = np.fft.fft(epoch, n_samples)
        psd = np.abs(fft_values) ** 2 / n_samples
        freqs = np.fft.fftfreq(n_samples, 1.0 / 256)  # Assuming a sampling rate of 256 Hz

        return psd, freqs

    def extract_features(self, preprocessed_data, n_components=9):
        # Ensure _preprocess_data() has been called before extracting features
        if self.scaled_features is None:
            raise ValueError("Data has not been preprocessed. Call _preprocess_data() first.")

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(preprocessed_data)

        return principal_components

    def apply_pca(self, n_components=3):
        pca = PCA(n_components=n_components)
        pca.fit(self.scaled_features)
        x_pca = pca.transform(self.scaled_features)
        return x_pca


    def _preprocess_data(self):
        #this finction making data ready for further analysis or machine learning tasks
        #by preventing some features from dominating others during analysis or modeling
        #using Standardization :subtracting the mean and dividing by the standard deviation
        self.scaled_features = self.signal_preprocessing(self.features)
        self.scaler.fit(self.scaled_features)
        self.scaled_features = self.scaler.transform(self.scaled_features)


    def split_data(self, test_size=0.33, random_state=42):
        return train_test_split(self.scaled_features, self.target, test_size=test_size, random_state=random_state)

    def svm_classifier(self, X_train, X_test, y_train, y_test):
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')  # Use RBF kernel and default C, gamma values
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("SVM Classifier:")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    def knn_classifier(self, X_train, X_test, y_train, y_test, neighbors=1):
        clf = KNeighborsClassifier(n_neighbors=neighbors)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print(f"KNN Classifier (k={neighbors}):")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

        #################################################################################
        sleep_detected = False  # Track if "Sleep" has been printed
        for prediction in predictions:
            if prediction == 1:
                self.consecutive_ones_count += 1
                if self.consecutive_ones_count >= 9 and not sleep_detected:
                    print("Sleep")
                    arduino.write(b'1')  # Send signal to turn on alarm
                    sleep_detected = True  # Set the flag to True
            else:
                self.consecutive_ones_count = 0
                sleep_detected = False  # Reset the flag

    #################################################################################





class EEGClassifierUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = EEGClassifier()
        self.ui.setupUi(self)
        self.ui.plot_all_channels_data()
        self.ui.plot_filtered_signals()  # Call plot_filtered_signal here



        X_train, X_test, y_train, y_test = self.ui.split_data()
        self.ui.knn_classifier(X_train, X_test, y_train, y_test, neighbors=1)
        X_train, X_test, y_train, y_test = eeg_classifier.split_data()
        eeg_classifier.svm_classifier(X_train, X_test, y_train, y_test)



if __name__ == "__main__":
    try:
        eeg_classifier = EEGClassifier()
        eeg_classifier._preprocess_data()
        preprocessed_data = eeg_classifier.scaled_features  # Get the preprocessed data
        features = eeg_classifier.extract_features(preprocessed_data)  # Pass preprocessed data to extract_features
        # Call preprocessing function
        app = QtWidgets.QApplication(sys.argv)
        window = EEGClassifierUI()
        window.show()
        sys.exit(app.exec_())

    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")