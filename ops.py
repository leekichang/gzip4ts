import numpy as np

from utils import class_repr

def moving_average(data, window_size):
    """
    Moving Average를 계산하는 함수
    
    Parameters:
        data (numpy.ndarray): 1차원 시계열 데이터
        window_size (int): 이동 평균을 계산하기 위한 윈도우 크기
    
    Returns:
        numpy.ndarray: Moving Average로 smoothing된 데이터
    """
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'same')
    
    return smoothed_data

def calculate_derivative(data):
    """
    1차원 데이터의 미분값을 계산하는 함수
    
    Parameters:
        data (numpy.ndarray): 1차원 시계열 데이터
    
    Returns:
        numpy.ndarray: 미분값으로 계산된 데이터 (원본 데이터 크기보다 1 작음)
    """
    # 이웃한 두 데이터 간의 차이를 계산하여 미분값 구하기
    derivative = np.diff(data)
    
    return derivative

def fft(data):
    """
    fft 계산하는 함수
    
    Parameters:
        data (numpy.ndarray): 1차원 시계열 데이터 (n_data, n_channel, ts_length)
    
    Returns:
        ?
    """
    
    
    fft_result = np.fft.fft(data, axis=2)
    
    sampling_rate = 320
    num_data = data.shape[0]
    num_channels = data.shape[1]
    num_points = data.shape[-1]
    
    
    frequencies = np.fft.rfftfreq(num_points, d=1/sampling_rate)  # Returns the frequencies
    # print(frequencies.shape, num_points, fft_result.shape)
    
    frq_pass = 700
    amplitude_spectrum = np.abs(fft_result)[:, :, 0:frq_pass]  # Normalize by the number of points
    
    phase_ang = (np.angle(fft_result))[:, :, :frq_pass] # TODO:이친구만의 푸리에를 찾자
    
    # phase_amplitude = np.abs(np.fft.fft(phase_ang, axis=2))
    
    # amplitude_spectrum 

    # import matplotlib.pyplot as plt
    
    # for data_idx in range(num_data):
        
    #     fig, axs = plt.subplots(num_channels, 3, figsize=(20, 6))
    #     if num_channels == 1:
    #         axs = [axs]
        
    #     for channel_idx in range(num_channels):
    #         ax = axs[channel_idx][0]
    #         ax.plot(data[data_idx][channel_idx])
            
    #         if channel_idx == 0:
    #             ax.set_title(f"Time domain data(class=)")
    #         elif channel_idx == num_channels -1:
    #             ax.set_xlabel("time")
    #         ax.set_ylabel(f"Channel {channel_idx}")
            
    #         ax = axs[channel_idx][1]
    #         ax.stem(amplitude_spectrum[data_idx][channel_idx])
    #         ax.set_yscale("log")
            
    #         if channel_idx == 0:
    #             ax.set_title("Amplitude spectrum")
    #         elif channel_idx == num_channels -1:
    #             ax.set_xlabel("Hz")
                
            
    #         if channel_idx == 0:
    #             ax.set_title("Magnitude spectrum")
    #         elif channel_idx == num_channels -1:
    #             ax.set_xlabel("Hz")
                
    #         ax = axs[channel_idx][2]
    #         ax.stem(phase_ang[data_idx][channel_idx])
                
    #         if channel_idx == 0:
    #             ax.set_title("Phase spectrum")
    #         elif channel_idx == num_channels -1:
    #             ax.set_xlabel("Hz")
            
                
    #     plt.tight_layout()
    #     plt.savefig(f"./plots/{data_idx}.png")
    #     plt.close()
    return amplitude_spectrum
    return np.concatenate([amplitude_spectrum, phase_ang], axis=1)