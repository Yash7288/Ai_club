data = readmatrix('samples.csv');
t = data(:,1);
x = data(:,2);
Ts = t(2) - t(1);    
fs = 1/Ts;          
N  = length(x);  
X = fft(x);
X_mag = abs(X)/N;
f = (0:N-1)*(fs/N);
figure
plot(f, X_mag)
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('FFT Magnitude Spectrum (Full Frequency Axis)')
grid on
