import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, HTML
from io import BytesIO
import open_clip
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import shap

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

class InterpretableImageClassifier(object):
    
    def __init__(self,
                 class_domain:str,
                 class_labels_list:list,
                 visual_clues_embeddings:dict,
                 image_dir:str, annotation_dir=None,
                 class_label_folders=False,
                 data_model_path=None
                ):
        
        self.class_domain = class_domain
        self.class_labels_list = class_labels_list
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.visual_clues_embeddings = visual_clues_embeddings
        self.data_model_path = data_model_path
        
        if class_label_folders:
            self.class_label_folder = [folder for folder in os.listdir(self.image_dir) if any(class_label in folder for class_label in self.class_labels_list)]
        
    def create_data_model_df(self):
        
        if os.path.exists(self.data_model_path):
            print(f"The file '{self.data_model_path}' already exists.")
            self.data = pd.read_csv(self.data_model_path)
        else:
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
                    
            if self.data_model_path:
                self.data.to_csv(self.data_model_path, index=False)
                
        return self.data
    
    def fit(self,
            print_model_perf_metrics=True,
            grid_search_params=None,
            select_most_important_features=False
           ):
        
        self.X = self.data
        self.y = self.data['class_label']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        
        # we keep a X_test_label dataframe to make some further analysis on correct and wrong predictions
        self.X_train = self.X_train.drop(columns=['class_label', 'image_name'])
        self.X_test_label = self.X_test.copy()
        self.X_test = self.X_test.drop(columns=['class_label', 'image_name'])
        
        if select_most_important_features:
            self.get_most_important_features()
            self.X_train = self.X_train[self.most_important_features_list]
            self.X_test = self.X_test[self.most_important_features_list]
        
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        if grid_search_params:
            
            grid_search = GridSearchCV(estimator=self.model, param_grid=grid_search_params, cv=3, verbose=2, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(self.X_train, self.y_train)
        
        self.y_pred = self.model.predict(self.X_test)
        self.y_score = self.model.predict_proba(self.X_test)
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(self.y_test, self.y_pred))
        
        self.X_test_label['pred'] = self.y_pred
        self.X_test_label = self.X_test_label[['class_label', 'image_name', 'pred']]
        
    def get_most_important_features(self,
                                    method='permutation',
                                    plot_results=True
                                   ):
        # we calculate feature importances using permutation method
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        feature_names = list(self.X_train.columns)
        perm_importance = permutation_importance(model, self.X_test, self.y_test, n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": perm_importance.importances_mean,
            "Importance Std": perm_importance.importances_std
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        # we do a grid search to find the most optimal number of features to use
        results = []
        feature_importance_thresholds_list = [-0.004, -0.003, -0.002, -0.001, 0, 0.001, 0.002, 0.003]
        for threshold in feature_importance_thresholds_list:
            
            selected_features = list(perm_importance_df[perm_importance_df.Importance > threshold].Feature)
            n_features = len(selected_features)
            X_selected = self.X[selected_features]
            
            X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, self.y, test_size=0.2, random_state=42)
            model_subset = RandomForestClassifier(random_state=42, n_jobs=-1)
            model_subset.fit(X_train_selected, y_train)
            
            y_pred_selected = model_subset.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred_selected)
            
            results.append((threshold, accuracy, n_features))
            
        best_threshold, best_score, best_n_features = max(results, key=lambda x: x[1])
        
        print(f"Best Threshold: {best_threshold}")
        print(f"Best Cross-Validation Accuracy: {best_score}")
        print(f"Number of Features Selected: {best_n_features}")
        
        thresholds, scores, n_features = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores, label='Cross-Validation Accuracy', marker='o')
        plt.xlabel('Feature Importance Threshold')
        plt.ylabel('Accuracy')
        plt.title('Grid Search for Feature Importance Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # we keep the the features that maximizes the accuracy of the model
        self.most_important_features_list = list(perm_importance_df[:best_n_features]['Feature'])
    
    def calculate_feature_importance(self, method):
        
        # Gini
        if method == 'gini':
            importances = self.model.feature_importances_
            feature_names = list(self.X_train.columns)
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
        
        # Permutation importance
        elif method == 'permutation':
            perm_importance = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=42)
            
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": perm_importance.importances_mean,
                "Importance Std": perm_importance.importances_std
            }).sort_values(by="Importance", ascending=False)
            
        else:
            raise ValueError("Feature importance method should be either 'gini' or 'permutation")
        
        return importance_df
        
    
    def predict(self,image_name, output_path=None):
        image_path = os.path.join(self.image_dir, image_name+'.jpg')
        annotation_path = os.path.join(self.annotation_dir, image_name)
        
        bounding_boxes = parse_annotation(annotation_path)
        image = Image.open(image_path)
        cropped_image = image.crop(bounding_boxes[0])
    
        image_text_similarities = get_image_cosine_similarities(image_path=image_path,
                                                                annotation_path=annotation_path,
                                                                text_embeddings=self.visual_clues_embeddings)
        
        df_pred = pd.DataFrame.from_dict(image_text_similarities, orient='index').T
        df_pred = df_pred[self.X_test.columns]
        
        print_text = f"\nthe true class_label is : {image_name.split('/')[0].split('-')[1]}\nthe predicted class_label is : {self.model.predict(df_pred)[0]}"
        print(print_text)
        
        probas = list(self.model.predict_proba(df_pred)[0])
        class_labels = list(self.model.classes_)
        prediction_probas = dict(zip(class_labels, probas))
        sorted_prediction_probas = dict(sorted(prediction_probas.items(), key=lambda item: item[1], reverse=False))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) #gridspec_kw={'width_ratios': [1.5, 1]}
                                       
        ax1.imshow(cropped_image)
        ax1.axis('off')
        
        ax2.barh(sorted_prediction_probas.keys(), sorted_prediction_probas.values())
        ax2.set_title('Model output probabilities for each class_label')
        
        plt.tight_layout()
        
        if output_path:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plot_image = Image.open(buffer)
            
            text_height = 100 
            combined_image = Image.new("RGB", (plot_image.width, plot_image.height + text_height), "white")
            combined_image.paste(plot_image, (0, 0))
            
            draw = ImageDraw.Draw(combined_image)
            text_position = (10, plot_image.height + 10)
            text_color = (0, 0, 0)  # Black text

            draw.text(text_position, print_text, fill=text_color)
            combined_image.save(output_path)