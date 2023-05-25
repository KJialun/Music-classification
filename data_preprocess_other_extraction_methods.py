import json
import os
import math
import librosa
import matplotlib.pyplot as plt
import numpy as np
DATASET_PATH = r"C:\Users\mfati\Dropbox\VUB 2022-2023\Machine Learning\MUSIC GENRE CLASSIFICATION\data"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
plott = 0

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along with genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs and labels
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        #which will increase the number of training data as 100 per genre is not much
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],  # they represent the targets or outputs we expect
        "mfcc": []  # they represent the training inputs
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment


                    # Depending on the extraction method to be tested, uncomment a line below.

                    #mfcc = librosa.feature.chroma_stft(y=signal[start:finish], sr=sample_rate)
                    #mfcc = librosa.feature.chroma_cqt(y=signal[start:finish], sr=sample_rate, n_chroma=36)
                    #mfcc = librosa.feature.chroma_cens(y=signal[start:finish], sr=sample_rate)
                    #mfcc = librosa.feature.chroma_vqt(y=signal[start:finish], sr=sample_rate, intervals='ji5',bins_per_octave=12)

                    #tempogram = librosa.feature.tempogram(y=signal[start:finish], sr=sample_rate)
                    #mfcc = librosa.feature.tempogram_ratio(tg=tempogram, sr=sample_rate)

                    #mfcc = librosa.feature.rms(y=signal[start:finish])
                    #mfcc = librosa.feature.spectral_centroid(y=signal[start:finish], sr=sample_rate)/hop_length
                    #mfcc = librosa.feature.spectral_bandwidth(y=signal[start:finish], sr=sample_rate)/hop_length
                    #mfcc = librosa.feature.spectral_contrast(y=signal[start:finish], sr=sample_rate)
                    #mfcc = librosa.feature.spectral_flatness(y=signal[start:finish])
                    # mfcc = librosa.feature.spectral_rolloff(y=signal[start:finish], sr=sample_rate)/hop_length
                    #mfcc = librosa.feature.poly_features(y=signal[start:finish], sr=sample_rate)
                    #mfcc = librosa.feature.tonnetz(y=signal[start:finish], sr=sample_rate)
                    #mfcc = librosa.feature.zero_crossing_rate(y=signal[start:finish])
                    #mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)