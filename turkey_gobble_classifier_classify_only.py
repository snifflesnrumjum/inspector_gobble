from keras.models import Model, load_model
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
from PIL import Image
import pandas as pd

###########
#paths for input/output

#this will be the path to the folder holding the WAV files
data_in = 'E:/CJunk/Turkey_gobble_identification/Darren_Data/test/' 

#this will be the path where you want the result files stored
data_out = 'E:/CJunk/Turkey_gobble_identification/outfiles/'

#what should the output files have appended to them?
outfile_suffix = '_CNN_results'

#where should the audio snippets go?
audio_out = 'E:/CJunk/Turkey_gobble_identification/Darren_Data/audio_out/' 

#where is the model?
model_path = 'E:/CJunk/Turkey_gobble_identification/turkey_classifier/model_16Dec2019_35_epochs.mdl' 
#############


############
#batch version of this function
def load_classify_images(infile):
    image_size_x = 125
    image_size_y = 49
    stft_params = {'nperseg':1000, 'noverlap':500}
    stft_conversion = stft_params['nperseg'] - stft_params['noverlap']
    window_shift_size = 500 #6000 equal 0.25seconds; 12000
    window_shift_size = int(window_shift_size / stft_conversion) #convert the window shift into stft units
    inwav = wavfile.read(infile) #load the raw WAV file
    outdata = pd.DataFrame(columns=['Event', 'Start_time', 'End_time', 'Gobble', 'Model_output', 'Min:Sec', 'Start_index', 'End_index', 'Num_windows', 'Begin File'])
    infilename = infile.split('/')[-1]
    gobble_count = 1
    results = []
    track_gobble = False
    gobble_start = 0
    gobble_end = 24000
    batch_results = {'start':[], 'end':[], 'result':[]}
    img_batch = []
    f, t, Zxx = signal.stft(inwav[1], 24000, nperseg=stft_params['nperseg'], noverlap=stft_params['noverlap']) #calculate the short time Fourier transform
    Zxx = np.abs(Zxx)[25:150, :]     
    
    #generate the window slices and run them through the CNN
    for image in range(0, int(Zxx.shape[1]), window_shift_size): #go through the WAV file in 1/12th second intervals (2000)
#         if image % 10000 == 0:
#             print(image)
        start_ind = image
        end_ind = image + 48
        
        if end_ind + 48 >= int(Zxx.shape[1]): #reached the end of the WAV file; no more 2s windows available
            continue
        else:
            batch_results['start'].append(start_ind)
            batch_results['end'].append(end_ind)
            img = Zxx[:, start_ind:start_ind+image_size_y].astype('int16') #slice the resulting array down to the proper size for input into the CNN
            img = img.reshape(image_size_x,image_size_y,1) 
            img = img/255.
            img_batch.append(img)
            if len(img_batch) == 16:
                out = model.predict(np.array(img_batch)) #run the images through the CNN
                for indiv_result in out:
                    batch_results['result'].append(indiv_result)
                img_batch = []
    
    if len(img_batch) > 0: #pickup the last few windows in case it doesn't end on a clean multiple of 16
        out = model.predict(np.array(img_batch)) #run the images through the CNN
        for indiv_result in out:
            batch_results['result'].append(indiv_result)
        img_batch = []
        
    #aggregate the results of the CNN    
    for result in range(len(batch_results['start'])):
        start_ind = batch_results['start'][result] * stft_conversion #convert back to WAV files units
        end_ind = batch_results['end'][result] * stft_conversion #convert back to WAV files units
        out = batch_results['result'][result][0]
        #if round(out) == 0 and not track_gobble: #CNN said it's a gobble; start tracking the length of the gobble window
        if out < 0.4 and not track_gobble: #CNN said it's a gobble; start tracking the length of the gobble window
            track_gobble = True
            gobble_start = start_ind
            gobble_end = start_ind + 24000
            gobble_predict_value = out
            num_gobble_windows = 1
        #elif round(out) == 0: #CNN said this window is still part of the previous gobble
        elif out < 0.5 and track_gobble: #CNN said this window is still part of the previous gobble
            #if start_ind > gobble_end:
            gobble_end += int(window_shift_size * stft_conversion) #extend the window size of the gobble
            gobble_predict_value += out
            num_gobble_windows += 1
        elif track_gobble == True: #CNN said the current window is no longer a gobble; complete the logging for this gobble
            track_gobble = False
            if num_gobble_windows > 16 and gobble_predict_value/num_gobble_windows < 0.5: #gobble window must be more than 2 windows AND the average value coming from the CNN less than 0.10
                gob_st = (gobble_start/24000) / 60.
                min_sec = '{0}:{1:02}'.format(int(gob_st), round(((gob_st) - int(gob_st))*60))
                outdata.loc[gobble_count] = [gobble_count, round(gobble_start/24000,2), round(gobble_end/24000,2), 1,
                                             round(gobble_predict_value/num_gobble_windows,4), min_sec, gobble_start, gobble_end, num_gobble_windows, infilename] #add the info to the DataFrame
                gobble_count += 1
                while gobble_end - gobble_start < 96000:
                    gobble_start -= 1000
                    gobble_end += 1000
                    if gobble_start < 0:
                        gobble_start = 0
                        break
                    
                outwav = inwav[1][gobble_start:gobble_end]
                wavfile.write(audio_out + infilename[:-4] + '_gobble_' + str(gobble_count - 1) + '.wav', 24000, outwav) #write the sound snippet to file
        else: #no gobble and it wasn't tracking a gobble
            pass 
        
    return outdata, batch_results['result']
#############


#start processing data below here   
model = load_model(model_path)
filelist = os.listdir(data_in) 
input_files = [x for x in os.listdir(data_in) if x[-3:] == 'wav']

for infile in input_files:
    print('Processing file ', infile, '...', end='')
    predictions, pred_scores = load_classify_images(data_in+infile)
    predictions.to_csv(data_out+infile[:-4]+outfile_suffix + '.txt', sep='\t')
    print('done!')