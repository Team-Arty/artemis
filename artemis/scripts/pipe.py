# Taking samples from the wikiart dataset
import pandas as pd
import pickle
from PIL import Image
import os
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
def pkl_op_creation(sample,path_persist_fol):
    # If sample button is clicked
    if(sample):
        number_of_samples = 6
        df= pd.read_csv(path_persist_fol+"/caption.csv")
        df=df.drop(["art_style","repetition"],axis=1)


        df.columns=['image_file','emotion','utterance']

        # Uncomment the following two lines if the emotion is not to be predicted by the model but is instead given as an input
        # df.columns=['image_file','grounding_emotion','utterance']
        # df['emotion']= df['grounding_emotion']

        df['image_file'] = path_persist_fol + '/wiki/Images/' + df['image_file'].astype(str) + '.jpg'
       
        df=df.sample(n=number_of_samples)
        df.to_csv(path_persist_fol + "/caption_test.csv")

        # Running a terminal operation 
        # Change the path to the respective files
        os.system("python3 /home/niegil/artemis/artemis/scripts/sample_speaker.py -speaker-saved-args /home/niegil/artemis/artemis/data/03-17-2021-20-32-19/config.json.txt -speaker-checkpoint /home/niegil/artemis/artemis/data/03-17-2021-20-32-19/checkpoints/best_model.pt -img-dir /home/niegil/artemis/artemis/data/wiki/Images -out-file /home/niegil/artemis/artemis/data/outputs/results.pkl --custom-data-csv /home/niegil/artemis/artemis/data/caption_test.csv") 
    
    else:

        df= pd.DataFrame()
        df['image_file'] = pd.Series([path_persist_fol + '/input.jpg'])
        df.to_csv(path_persist_fol + "/caption_test.csv")

        # Running a terminal operation
        # Change the path to the respective files
        os.system("python3 /home/niegil/artemis/artemis/scripts/sample_speaker.py -speaker-saved-args /home/niegil/artemis/artemis/data/03-17-2021-20-32-19/config.json.txt -speaker-checkpoint /home/niegil/artemis/artemis/data/03-17-2021-20-32-19/checkpoints/best_model.pt -img-dir /home/niegil/artemis/artemis/data/wiki/Images -out-file /home/niegil/artemis/artemis/data/outputs/results.pkl --custom-data-csv /home/niegil/artemis/artemis/data/caption_test.csv  --img2emo-checkpoint /home/niegil/artemis/artemis/data/best_model.pt")
        

# Pass True to sample if the images need to be sampled from wikiart
# If an image is uploaded - the image should be named input.jpg within the persistent folder 
def predict(sample=False):
    path_persist_fol = "/home/niegil/artemis/artemis/data"
    pkl_op_creation(sample,path_persist_fol)
    objects = []
    with (open(path_persist_fol + '/outputs/results.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df=objects[1][0][1]

    pred_dict = {}
    pred_dict["img_name"] = []
    pred_dict["file_loc"] = []
    pred_dict["emotion_pred"] = []
    pred_dict["caption_pred"] = []

    for i,j,k in zip(df['image_file'],df['grounding_emotion'],df['caption']):
        im = Image.open(i).resize((256,256))
        display(im)
        pred_dict["img_name"].append(i.split('/')[-1].split('.')[0])
        pred_dict["file_loc"].append(i)
        pred_dict["emotion_pred"].append(j)
        pred_dict["caption_pred"].append(k)

        print('Image:', i.split('/')[-1].split('.')[0])
        print('Emotion Predicted :',j)
        print('Caption Predicted :',k)
        print('--------------------------------------------------------------------------------------')
    return pred_dict

predict()