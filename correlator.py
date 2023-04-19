import numpy as np
from scipy.fftpack import fft,ifft
import os
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import math

def filt(y,down,up):    # a function to filter the spectrum
    fft_y=fft(y)
    datalength=fft_y.shape[0]//2
    fft_y[0:down]=0
    fft_y[up:2*datalength-up]=0
    fft_y[2*datalength-down:]=0
    return fft_y

def compare(cross,name):   # a simple funtion to plot the different features of a complex function, picture saved as the name we give
    fig,ax=plt.subplots(3,1,sharex='col')
    x=np.array(range(cross.shape[0]))
    ax1=ax[0]
    ax1.plot(x,np.abs(cross))   # plot the absolute value
    ax1.set_title('abs')
    ax2=ax[1]
    ax2.plot(x,np.real(cross))   # plot the real part of the function
    ax2.set_title('real')
    ax3=ax[2]
    ax3.plot(x,np.angle(cross))    # plot the phase angle of the function
    ax3.set_title('angle')
    plt.savefig(name)
    plt.close()
    return 0

def move(y,m):  # a simple funtion to shift an array by a given length. we may use this function if we want to try incoherent dedispersion
    x=np.empty(y.shape[0],dtype=complex)
    if m<0:
        x[0:y.shape[0]+m]=y[-m:]
        x[y.shape[0]+m:]=y[0:-m]
    else:
        x[0:m]=y[y.shape[0]-m:]
        x[m:]=y[0:y.shape[0]-m]
    return x
    
def generate_spectrum(data,timestep):    # generate the spectrum of a time-domain signal
    datalength=data.shape[0]
    spectrum=np.empty([datalength//timestep,timestep],dtype=complex)   # allocate storage for the calculation
    count=datalength//timestep
    for i in range(count):
        shortspec=fft(data[i*timestep:(i+1)*timestep])   # do STFT of the signal 
        spectrum[i]=shortspec
    spectrum=np.array(spectrum)
    return spectrum
    
def plot_spectrum(spectrum,downfre,upfre,fre,name):    # plot the spectrum we have as a waterfall plot
    up=(upfre*spectrum.shape[0])//fre
    down=(downfre*spectrum.shape[0])//fre
    spectrum=np.abs(spectrum)**2
    print(spectrum.shape)    
    spectrum=spectrum[down:up,:]
    b=np.abs(spectrum)+3*np.std(spectrum)
    a=np.abs(spectrum)-3*np.std(spectrum)
    plt.imshow(spectrum,vmin=a[0,0],vmax=b[0,0])
    plt.savefig(name)
    plt.close()
    
def de_noise(spectrum,downfre,upfre,fre):  # subtract the noise from the signal between two certain frequencies
    long_noise=np.mean(spectrum,axis=0)
    long_noise=np.vstack([long_noise]*spectrum.shape[0])   # culcalate the long-time noise
    spectrum=spectrum-long_noise   # subtract long-time noise
    up=(upfre*spectrum.shape[1])//fre
    down=(downfre*spectrum.shape[1])//fre
    datalength=spectrum.shape[1]//2
    up=int(up)
    down=int(down)
    noise=[]
    for i in range(down,up):
        timeline=np.abs(spectrum[:,i])**2
        M=timeline.shape[0]
        SK=M/(M-1)*(((M+1)*np.sum(timeline**2))/(np.sum(timeline)**2)-2)   # culculate the spectral kurtosis
        if SK>0.02 or SK<-0.02:   # mostly, astronomical signals contribute a spectral kurtosis as 0, so those none-zero channels contain noise
            noise.append(i)
    for i in range(2*datalength-up,2*datalength-down):
        timeline=np.abs(spectrum[:,i])**2
        M=timeline.shape[0]
        SK=M/(M-1)*(((M+1)*np.sum(timeline**2))/(np.sum(timeline)**2)-2)  # do the same thing for the other part of spectrum
        if SK>0.02 or SK<-0.02:
            noise.append(i)
    spectrum[:,noise]=0    # make the channels with short-time noise zero, which means they won't be considered in later calculation
    spectrum[:,0:down]=0   # make the channels beyond our frequency range 0
    spectrum[:,up:2*datalength-up]=0
    spectrum[:,2*datalength-down:]=0
    print(noise)
    return spectrum

def dedisperse(data,fre,DM,fre0):   # do dedispersion for the signal
    spectrum=fft(data)   # generate a spectrum
    datalength=spectrum.shape[0]//2
    deltat=range(datalength)
    deltat=np.array(deltat)
    deltat=deltat*fre/(spectrum.shape[0])+fre0    # culculate the frequency for every point in the spectrum
    phase=4.15*10**15*(deltat**(-2)-fre0**(-2))*DM*deltat    # culculate the phase caused by the time-delay
    spectrum[0:datalength]=spectrum[0:datalength]*np.exp(-2J*phase*math.pi)   # correct the phase of the spectrum
    spectrum[datalength:2*datalength]=np.conjugate(np.flipud(spectrum[0:datalength])) # fulfill the other half of spectrum by conjugate symmetry
    print('dedisperse completed')
    return spectrum

def make_bin(data,a):   # culculate bins of the data by a certain length a
    datalength=data.shape[0]//a
    bin=np.empty(datalength)
    for i in range(datalength):
        bin[i]=np.sum(np.abs(data[i*a:(i+1)*a])**2) # add up the square of data to gain the intensity
    return bin

def back_to_time(spectrum):     # convert the spectrum from frequency domain back to time domain
    data=np.empty(spectrum.shape[0]*spectrum.shape[1],dtype=complex)  # allocate storage
    length=spectrum.shape[1]
    for i in range(spectrum.shape[0]):
        data[i*length:(i+1)*length]=ifft(spectrum[i,:])   # reverse the DTFT
    print('back_to_time completed')
    return data
    
def find_pulse(data,k,bintime):   # find one pulse in a length of data
    intensity=make_bin(data,a=bintime)
    x0=np.mean(intensity)
    sigma=np.std(intensity)   # culculate the mean value and standard deviation
    if (intensity[intensity>x0+5*sigma].shape[0])<5:   # we believe the points which are larger than 5 sigma are part of the pulse
        return False
    else:
        position=np.argmax(intensity)
        t=np.arange(intensity.shape[0])
        intensity=intensity/sigma   # normalize data by the standard deviation
        plt.plot(t,intensity)
        plt.title('(after correlation) data number '+str(k)+' with max index at '+str(position))
        plt.savefig('pulses/(after correlation) data number '+str(k)+' with max index at '+str(position)+'.jpg')
        plt.close()
        i=make_bin(data[position*bintime-5000000:position*bintime+5000000],a=10)
        i=i/np.std(i)
        plt.plot(i)
        plt.savefig('pulses/(after correlation) pulse zoomed.jpg')
        plt.close()
        i=make_bin(data[position*bintime-200000:position*bintime+200000],a=10)
        i=i/np.std(i)
        plt.plot(i)
        plt.savefig('pulses/(after correlation) pulse more zoomed.jpg')
        plt.close()    # plot the profile of the pulse at various scales
        return k,position

def fft_interference(data1,data2,fre,offset,step):   # calculate the interference fringe by the method of FX correlator
    if data1.shape != data2.shape:   # make sure the calculation is right
        return False
    spec1=fft(data1)   # do fft to yield spectrum
    spec2=fft(data2)
    cross=spec1*np.conjugate(spec2)   # do correlation
    cross=ifft(cross)   # convert the fringe back to time domain
    c=np.abs(cross)
    compare(cross,name='all_after.jpg')   # plot the fringe
    compare(cross[0:500],name='head_after.jpg')
    compare(cross[cross.shape[0]-500:],name='tail_after'+str(np.argmax(c))+'.jpg')
    compare(cross[300000000:300000500],name='middle_after.jpg')
    return 0

def monotone(a):   # a function to make the phase of an array monotonous
    k=0
    d=2*np.pi   # the cycle of phase is 2pi
    b=a[0]
    for i in range(a.shape[0]):
        if a[i]==0:
            pass
        elif abs(a[i]-b)>np.pi:
            k+=d
            b=a[i]
            a[i]+=k
        else:
            b=a[i]
            a[i]+=k
    return a

def spectrum_fitting(spectrum1,spectrum2,fre,fre0,name):   # fit the phase to yield time delay
    if spectrum1.shape != spectrum2.shape:   # make sure the calculation is right
        return False
    cross=spectrum1*np.conjugate(spectrum2)   # do correlation
    phase=np.angle(cross)  # get the phase of the fringe
    #plt.imshow(phase)
    #plt.colorbar()
    #plt.savefig('phase of spectrum_0.jpg')
    #plt.close()
    f=np.array(range(spectrum1.shape[1]))   # allocate storage
    f=f*fre/f.shape[0]+fre0    # f can represent the frequency of the spectrum
    l=f.shape[0]//2
    time=np.empty(spectrum1.shape[0])
    b=np.empty(spectrum1.shape[0])    # allocate storage to save values of every time window
    for i in range(spectrum1.shape[0]):
        p=np.unwrap(phase[i,:])   # make the phase monotone increasing
        eff=np.nonzero(p[0:l])    # only consider the channels without noise
        t=np.polyfit(f[eff],p[eff],deg=1)   # fit the phase to yield timedelay
        time[i]=t[0]/(2*np.pi)  # save the timedelay
        b[i]=t[1]   # save the intercept of the fitting
    x=np.array(range(spectrum1.shape[0]))
    fig,ax=plt.subplots(2,1,sharex='col')
    ax1=ax[0]
    ax1.plot(x,time)
    ax1.set_title('time delay with mean time delay as '+str(np.mean(time)))
    ax2=ax[1]
    ax2.plot(x,b)
    ax2.set_title('b')
    plt.savefig(name+'all_fit.jpg')
    plt.close() # plot the time delay we have
    plt.scatter(f[0:l],p[0:l])
    plt.plot(f[0:l],t[0]*f[0:l]+t[1],c='red')
    plt.title('time='+str(t[0]/(2*np.pi))+' b='+str(t[1]))
    plt.savefig(name+'fit_0.jpg') # plot one fitting result to check if the fitting is reliable
    plt.close()
    return t

PATH0='/mnt/ssd0/power_-60dbm/'
number1=13835058074324629497
number2=13907115668362557567
num1=str(hex(number1))
num2=str(hex(number2))
num1=num1[4:18]   # the first 8 bits of the number represent the polarization, and other bits represent the number of packs
num2=num2[4:18]   # normally, data is recorded continuously, thus the difference of the starting number of pack represent the time delay when recording
num1=int(num1,16)
num2=int(num2,16)
filepath1=PATH0+str(number1)+'.bin'
filepath2=PATH0+str(number2)+'.bin'
size1=os.path.getsize(filepath1)
print(size1)
size2=os.path.getsize(filepath2)
print(size2)
#file0=open(filepath1,'rb')
#datalength=1000000000
#size=size1//datalength
#for i in range(size):
    #print(i)
    #data0=np.fromfile(file0,dtype=np.int8,count=datalength)
    #plt.specgram(data0,NFFT=2**16,Fs=1000000000)
    #plt.savefig('spectrum_pulse3_'+str(i)+'.jpg')
    #plt.close()
    #spec=generate_spectrum(data0,2**16)
    #k=de_noise(spec,0,500000000,1000000000)
    #plt.close()
    #k=dedisperse(k,0,500000000,1000000000,DM=-56.771)
    #a=find_pulse(k,i,bintime=2**16)
    #if a==False:
        #print('false')
    #else:
        #print(a)
file1=open(filepath1,'rb')
file2=open(filepath2,'rb')
datalength=900000000
timedelay=-4
if num1>=num2:
    file2.read((num1-num2)*4096)   # the time of each pack is 4096ns, and data is recorded 1 byte per ns
else:
    file1.read((num2-num1)*4096)
if timedelay<0:
    file1.read(-timedelay)
else:
    file2.read(timedelay)
data1=np.fromfile(file1,dtype=np.int8,count=datalength)
spectrum1=generate_spectrum(data1,2**14)
#plot_spectrum(spectrum1,downfre=0,upfre=5500000000,fre=1000000000,name='pulses/spectrum_before_correlation.jpg')
data2=np.fromfile(file2,dtype=np.int8,count=datalength)
spectrum2=generate_spectrum(data2,2**14)
#spectrum1=dedisperse(data1,1000000000,DM=56.771,fre0=1000000000)
file1.close()
file2.close()
#spectrum2=de_noise(spectrum2,downfre=0,upfre=500000000,fre=1000000000)
#data1=back_to_time(spectrum1)
#compare(spectrum1,'pulses/test_after.jpg')
#compare(data1,'pulses/test_data_correlated.jpg')
#k=find_pulse(data1,0,bintime=2**12)
#print('find_pulse completed')
#print(k)
#file0=open('log.txt','a')
#file0.write(str(k))
#file0.close()
#fft_interference(data1,data2,1000000000,0,0)
t=spectrum_fitting(spectrum1,spectrum2,fre=1000000000,fre0=1000000000,name='noise_test/60dm_')
cross=spectrum1*np.conjugate(spectrum2)
compare(cross[0,1:],'noise_test/60dm_fringe0.jpg')
compare(cross[0,1000:1050],'noise_test/60dm_fringe_zoom.jpg')
#np.savetxt('log.txt',t)
#138_9497 and 139_7567are directly connected to noise sourse
