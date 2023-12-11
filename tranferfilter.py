import numpy as np
import matplotlib.pyplot as plt

fc = 2  # Frekuensi cutoff
n = 2   # Orde filter
f = np.linspace(0, 10, 100)  # Membuat array frekuensi dari 0 hingga 10 Hz

H = 1 / np.sqrt(1 + (f / fc)**(2 * n))  # Menghitung fungsi transfer

plt.plot(f, H)
plt.title('Respon Frekuensi Filter Low-Pass Butterworth')
plt.xlabel('Frekuensi (Hz)')
plt.ylabel('Amplitudo')
plt.grid(True)
plt.show()