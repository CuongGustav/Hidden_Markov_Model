#trích xuất các đặc trưng MFCC từ tệp âm thanh
import math
import librosa
import numpy as np

def get_mfcc(file_path):
    y, sr = librosa.load(file_path)     # read .wav file 
                                        #sr: Tần số mẫu (sampling rate), thường là 22050 Hz mặc định.
    if len(y) == 0:
        raise ValueError("Audio data is empty after loading.")
        
    hop_length = math.floor(sr * 0.010)  # 10ms hop Số mẫu di chuyển giữa các khung.
    win_length = math.floor(sr * 0.025)  # 25ms frame

    # Kiểm tra để đảm bảo tín hiệu đủ dài cho n_fft
    if len(y) < 1024:
        raise ValueError("Audio signal is too short for the specified n_fft.")

    # mfcc is 13 x T matrix
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=11, n_fft=1024,
        hop_length=hop_length, win_length=win_length
    )  # Ensure y and sr are passed as keyword arguments
    
    # subtract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1, 1))    #làm nổi bật các biến đổi so với giá trị trung bình.
    
    # delta feature 1st order and 2nd order
    if mfcc.shape[1] < 9:
        raise ValueError(f"MFCC data too short with shape {mfcc.shape} for delta calculation.")
        
    delta1 = librosa.feature.delta(mfcc, order=1)   #Biến thiên bậc 1 (thay đổi theo thời gian) của MFCC.
    delta2 = librosa.feature.delta(mfcc, order=2)   #Biến thiên bậc 2 (biến đổi của delta1) của MFCC.
    
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0)  # O^r
    # return T x 36 (transpose of X)
    return X.T  # hmmlearn use T x N matrix
