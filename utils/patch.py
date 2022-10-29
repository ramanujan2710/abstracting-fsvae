import torch
import logging
from CLIPEmbedding import CLIPEmbedding
from skimage.io import imread
import argparse
import os
import glob

logging.basicConfig(filename= 'logs',level = logging.DEBUG)

#NOTE
# The pipeline is as follows:
# create dataset of images/text --> get the filenames (for images)/ the list of text (for text data) --> generate_model() [specify the path to  model checkpoint] --> embed()
# [embed needs indivudial data (for example single image or a text, for a directory, first get a list of all files then feed them iteratively)
#
# END

def get_args():
    
    #command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = "openai/clip-vit-base-patch32" , help = "path to embedding model") #if model checkpoint is available locally, specify
    parser.add_argument('--device',choices = ['cpu','cuda'],default = 'cpu', help = "device")
    parser.add_argument('--data', default = None, help = "text or path to image directory or a single image")
    parser.add_argument('--img_extension', default ='jpg', help = 'Extension of image files in the directory [data], if data is a directory')
    return parser.parse_args()
    

def generate_model(args):
	
	#generates the model for embeds	
	return CLIPEmbedding( model_path = args.model_path, device = args.device) #defaults to 'cpu')
	 
	
def embed(model, data = None):
    
    #embeds data (both text and image)
    # the data is a 'string' type, that can be the path to an image or simply a text
    # this function will determine if data is a text or image and produce the respective embedding
    assert data is not None, 'ERROR: No data'
     
    if os.path.isfile(data):
            
        #image is supplied as a path
        return model.embed_images([imread(data)]) 
    else:
            
        return model.embed_text([data])
        


def get_filenames(args):

    #list all file names that are of same extension    
    file_names = glob.glob(os.path.join(args.data, '**/*.'+ args.img_extension), recursive = True)
    return file_names

if __name__ == "__main__":
    
    # get the arguments and create the model
    # use the model to generate embeddings for the data
    args = get_args() 
    model = generate_model(args)
    
    if os.path.isdir(args.data):
        
        files = get_filenames(args)
        embeddings = []
        for file in files:
            
            embeddings.append(embed(model, data = file))
        
    else:
        
        embeddings = embed(model, data = args.data)
        
        
        
    print(embeddings)
    
    
    
    
    
