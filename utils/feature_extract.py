import numpy as np
from scipy.stats import kurtosis
from scipy.stats import entropy
import nolds
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
from scipy.stats import skew

def feature_extract(dataset):
    '''

    '''
    print(dataset.shape)
    new_dataset = []
    for i in range(dataset.shape[0]):
        print(i)
        feature = []
        # 提取信号
        data = dataset[i,0,0:1024]
        # 将NumPy数组转换为列表
        data = data.tolist()
        # 1. 峭度（Kurtosis）：衡量信号的尖锐程度，用于检测信号中的高频成分。
        kurt = kurtosis(data)
        feature.append(kurt)
        # 2. 熵值（Entropy）：衡量信号的复杂程度和随机性，用于检测信号的频谱特性。
        ent = entropy(data)#  -inf 表示  负无穷（-inf），则表示信号的熵为无穷大。这通常意味着信号具有非常高的不确定性和复杂性，其中的值没有明显的模式或规律
        if (ent<-0.000001 ):
            ent = 0
        feature.append(ent)
        # 3. 分形值（Fractal Dimension）：衡量信号的自相似性和复杂度，用于分析信号的分形特征。
        fd = nolds.dfa(data)
        feature.append(fd)
        # 4. 波形指标（Waveform Indicators）：峰值因子，用于分析信号的时域特征。
        peak_factor = np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data)))
        feature.append(peak_factor)
        # 5. 波形指标（Waveform Indicators）：脉冲因子，用于分析信号的时域特征。
        pulse_factor = np.max(np.abs(data)) / np.mean(np.abs(data))
        feature.append(pulse_factor)
        # 6. 波形指标（Waveform Indicators）：裕度因子，用于分析信号的时域特征。
        crest_factor = np.max(np.abs(data)) / np.mean(np.sqrt(np.mean(np.square(data))))
        feature.append(crest_factor)
        # 7. 频谱指标（Spectral Indicators）：能量比值，用于分析信号的频域特征。
        # 计算信号的频谱
        sampling_rate = 1024  # 采样率
        freq, power_spectrum = welch(data, fs=sampling_rate)
        # 计算峰值频率
        peak_freqs, _ = find_peaks(power_spectrum, height=np.mean(power_spectrum))  # 找到峰值
        # 计算能量比值
        total_energy = np.sum(power_spectrum)
        peak_energy = np.sum(power_spectrum[peak_freqs])
        energy_ratio = peak_energy / total_energy
        feature.append(energy_ratio)
        # 8. 频谱指标（Spectral Indicators）：谱线形指标，用于分析信号的频域特征。
        # 计算谱线形指标
        print(np.mean(power_spectrum))
        spectral_flatness = np.exp(np.mean(np.log(power_spectrum))) / (np.mean(power_spectrum))
        feature.append(spectral_flatness)
        # 9. 统计特征（Statistical Features）：均值，用于描述信号的统计特性。
        mean = np.mean(data)
        feature.append(mean)
        # 10. 统计特征（Statistical Features）：方差，用于描述信号的统计特性。
        variance = np.var(data)
        feature.append(variance)
        # 11. 统计特征（Statistical Features）：偏度，用于描述信号的统计特性。
        skewness = skew(data)
        feature.append(skewness)
        # 12. 振动特征（Vibration Features）：包括峰值振动、有效值振动等，用于描述信号的振动特性。
        peak_vibration = np.max(np.abs(data))
        feature.append(peak_vibration)
        # 13. 振动特征（Vibration Features）：包括峰值振动、有效值振动等，用于描述信号的振动特性。
        rms_vibration = np.sqrt(np.mean(np.square(data)))
        feature.append(rms_vibration)
        new_dataset.append(feature)

    new_dataset = np.array(new_dataset)
