import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_beats(beat_times, out_file_path="beat_times_2.csv"):
    # y_beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print('Saving output to %s' %(out_file_path))
    librosa.output.times_csv(out_file_path, beat_times)

def get_beat_times_from_song(file_path="data/audio_samples/Duke Ellington - That rhythm man.wav", is_save=False):
    y, sr = librosa.load(file_path)

    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    if is_save:
        save_beats(beat_times)

    return beat_times

def plot_beats_and_clicks(file_path):
    y, sr = librosa.load(file_path)
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    y_beat_times = librosa.clicks(times=times, sr=sr)


    plt.figure()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    ax = plt.subplot(2,1,2)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             x_axis='time', y_axis='mel')
    plt.subplot(2,1,1, sharex=ax)
    librosa.display.waveplot(y_beat_times, sr=sr, label='Beat clicks')
    plt.legend()
    plt.xlim(15, 30)
    plt.tight_layout()
    plt.show()

def mix_beats_and_clicks(file_path, out_file_path="test.wav"):
    y, sr = librosa.load('data/audio_samples/Duke Ellington - That rhythm man.wav')
    _, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
    clicks = librosa.clicks(beat_times, sr=sr, length=len(y))
    y_combined = y + clicks
    librosa.output.write_wav(out_file_path, y_combined, sr)

def main(file_path="data/audio_samples/Duke Ellington - That rhythm man.wav"):
    _ = get_beat_times_from_song()
    plot_beats_and_clicks(file_path)
    mix_beats_and_clicks(file_path)



if __name__ == "__main__":
    main()

# https://www.researchgate.net/publication/229076713_A_multi-modal_dance_corpus_for_research_into_interaction_between_humans_in_virtual_environments
#
#
# # approach 2 - dbn tracker
# proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
# act = madmom.features.beats.RNNBeatProcessor()('data/audio_samples/Duke Ellington - That rhythm man.wav')
#
# beat_times = proc(act)
#
# clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
# ipd.Audio(x + clicks, rate=sr)


# import wave, array, math, time, argparse, sys
# import numpy, pywt
# from scipy import signal
# import pdb
# import matplotlib.pyplot as plt
#
# def read_wav(filename):
#
#     #open file, get metadata for audio
#     try:
#         wf = wave.open(filename,'rb')
#     except IOError as e:
#         print(e)
#         return
#
#     # typ = choose_type( wf.getsampwidth() ) #TODO: implement choose_type
#     nsamps = wf.getnframes();
#     assert(nsamps > 0);
#
#     fs = wf.getframerate()
#     assert(fs > 0)
#
#     # read entire file and make into an array
#     samps = list(array.array('i',wf.readframes(nsamps)))
#     #print 'Read', nsamps,'samples from', filename
#     try:
#         assert(nsamps == len(samps))
#     except AssertionError as e:
#         print(nsamps, "not equal to", len(samps))
#
#     return samps, fs
#
# # print an error when no data can be found
# def no_audio_data():
#     print("No audio data for sample, skipping...")
#     return None, None
#
# # simple peak detection
# def peak_detect(data):
#     max_val = numpy.amax(abs(data))
#     peak_ndx = numpy.where(data==max_val)
#     if len(peak_ndx[0]) == 0: #if nothing found then the max must be negative
#         peak_ndx = numpy.where(data==-max_val)
#     return peak_ndx
#
# def bpm_detector(data,fs):
#     cA = []
#     cD = []
#     correl = []
#     cD_sum = []
#     levels = 4
#     max_decimation = 2**(levels-1);
#     min_ndx = 60./ 220 * (fs/max_decimation)
#     max_ndx = 60./ 40 * (fs/max_decimation)
#
#     for loop in range(0,levels):
#         cD = []
#         # 1) DWT
#         if loop == 0:
#             [cA,cD] = pywt.dwt(data,'db4');
#             cD_minlen = len(cD)/max_decimation+1;
#             cD_sum = numpy.zeros(cD_minlen);
#         else:
#             [cA,cD] = pywt.dwt(cA,'db4');
#         # 2) Filter
#         cD = signal.lfilter([0.01],[1 -0.99],cD);
#
#         # 4) Subtractargs.filename out the mean.
#
#         # 5) Decimate for reconstruction later.
#         cD = abs(cD[::(2**(levels-loop-1))]);
#         cD = cD - numpy.mean(cD);
#         # 6) Recombine the signal before ACF
#         #    essentially, each level I concatenate
#         #    the detail coefs (i.e. the HPF values)
#         #    to the beginning of the array
#         cD_sum = cD[0:cD_minlen] + cD_sum;
#
#     if [b for b in cA if b != 0.0] == []:
#         return no_audio_data()
#     # adding in the approximate data as well...
#     cA = signal.lfilter([0.01],[1 -0.99],cA);
#     cA = abs(cA);
#     cA = cA - numpy.mean(cA);
#     cD_sum = cA[0:cD_minlen] + cD_sum;
#
#     # ACF
#     correl = numpy.correlate(cD_sum,cD_sum,'full')
#
#     midpoint = len(correl) / 2
#     correl_midpoint_tmp = correl[midpoint:]
#     peak_ndx = peak_detect(correl_midpoint_tmp[int(min_ndx):int(max_ndx)]);
#     if len(peak_ndx) > 1:
#         return no_audio_data()
#
#     peak_ndx_adjusted = peak_ndx[0]+min_ndx;
#     bpm = 60./ peak_ndx_adjusted * (fs/max_decimation)
#     print(bpm)
#     return bpm,correl
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process .wav file to determine the Beats Per Minute.')
#     parser.add_argument('--filename', required=True,
#                    help='.wav file for processing')
#     parser.add_argument('--window', type=float, default=3,
#                    help='size of the the window (seconds) that will be scanned to determine the bpm.  Typically less than 10 seconds. [3]')
#
#     args = parser.parse_args()
#     samps,fs = read_wav(args.filename)
#
#     data = []
#     correl=[]
#     bpm = 0
#     n=0;
#     nsamps = len(samps)
#     window_samps = int(args.window*fs)
#     samps_ndx = 0;  #first sample in window_ndx
#     max_window_ndx = nsamps / window_samps;
#     bpms = numpy.zeros(max_window_ndx)
#
#     #iterate through all windows
#     for window_ndx in xrange(0,max_window_ndx):
#
#         #get a new set of samples
#         #print n,":",len(bpms),":",max_window_ndx,":",fs,":",nsamps,":",samps_ndx
#         data = samps[samps_ndx:samps_ndx+window_samps]
#         if not ((len(data) % window_samps) == 0):
#             raise AssertionError( str(len(data) ) )
#
#         bpm, correl_temp = bpm_detector(data,fs)
#         if bpm == None:
#             continue
#         bpms[window_ndx] = bpm
#         correl = correl_temp
#
#         #iterate at the end of the loop
#         samps_ndx = samps_ndx+window_samps;
#         n=n+1; #counter for debug...
#
#     bpm = numpy.median(bpms)
#     print('Completed.  Estimated Beats Per Minute:', bpm)
#
#     n = range(0,len(correl))
#     plt.plot(n,abs(correl));
#     plt.show(False); #plot non-blocking
#     time.sleep(10);
#     plt.close();
