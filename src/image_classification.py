import os
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import open_clip
import torch
import torch.nn.functional as F

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bboxes = []
    
    # Extract bounding box coordinates
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    
    return bboxes

def get_image_cosine_similarities(image_path:str, text_embeddings:dict, annotation_path=None): 
    # get the bounding boxes of the image to know where the important part is located on the image
    if annotation_path:
        bounding_boxes = parse_annotation(annotation_path)
        image = Image.open(image_path)
        image = image.crop(bounding_boxes[0])
    
    # get image embedding from CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    inputs = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        img_embedding = clip_model.encode_image(inputs)
    
    # calculate cosine similarity between the image and each visual clue sentence
    image_texts_similarities = {text: F.cosine_similarity(img_embedding, text_embedding, dim=-1).item() for text, text_embedding in text_embeddings.items()}
    
    return image_texts_similarities

class InterpretableImageClassification(object):
    
    def __init__(self, class_domain:str, class_labels_list:list, visual_clues_embeddings:dict, image_dir:str, annotation_dir=None, class_label_folders=False):
        self.class_domain = class_domain
        self.class_labels_list = class_labels_list
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.visual_clues_embeddings = visual_clues_embeddings
        
        if class_label_folders:
            self.class_label_folder = [folder for folder in os.listdir(self.image_dir) if any(class_label in folder for class_label in self.class_labels_list)]
        
    def create_model_data_df(self, data_path=None):
        
        self.data= pd.DataFrame(columns=['class_label', 'image_name']+ list(self.visual_clues_embeddings.keys()))
        
        for folder in self.class_label_folder:
            class_label = folder.split('-')[1]
            folder_path_images = os.path.join(self.image_dir, folder)
            images = [os.path.join(self.image_dir, folder, img) for img in os.listdir(folder_path_images) if '.ipynb_checkpoints' not in img]
            
            for image_path in images:
                image_name = image_path.split("/")[-1].replace(".jpg", "")
                if self.annotation_dir:
                    annotation_path = os.path.join(self.annotation_dir, folder,  image_name)
                
                image_text_similarities = get_image_cosine_similarities(image_path=image_path,
                                                                        text_embeddings=self.visual_clues_embeddings,
                                                                        annotation_path=annotation_path)
                image_text_similarities['class_label'] = class_label
                image_text_similarities['image_name'] = image_name
                self.data = pd.concat([self.data, pd.DataFrame.from_dict(image_text_similarities, orient='index').T])
                
        if data_path:
            self.data.to_csv(data_path, index=False)
                
        return self.data
    
    def fit():
        