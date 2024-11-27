import os
import re
import ast
import pydot
from IPython.display import Image, display
from langchain_openai import AzureChatOpenAI
import open_clip
import torch
import json
from dotenv import load_dotenv
load_dotenv()

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dict_to_pydot(graph, class_domain_dict, class_label):
    """Recursively add nodes and edges from a nested dictionary to a pydot graph."""
    for key, value in class_domain_dict.items():
        node_name = f"{class_label}_{key}"
        graph.add_node(pydot.Node(node_name, label=key))
        graph.add_edge(pydot.Edge(class_label, node_name))
        
        # If the value is a dictionary, recursively add its children
        if isinstance(value, dict):
            dict_to_pydot(graph, value, node_name)
        else:
            # Create a leaf node for non-dict values
            leaf_name = f"{node_name}_{value}"
            graph.add_node(pydot.Node(leaf_name, label=str(value)))
            graph.add_edge(pydot.Edge(node_name, leaf_name))


class VisualClues(object):
    
    def __init__(self, 
                 class_domain:str,
                 class_labels_list:list,
                 visual_clue_tree_path=None
                ):
        
        self.class_domain = class_domain
        self.class_labels_list = class_labels_list
        self.llm_model = AzureChatOpenAI(
            temperature=0,
            deployment_name=os.getenv('CHAT_DEPLOYEMENT_NAME'),
            model_name=os.getenv('CHAT_MODEL_NAME'),
            openai_api_version=os.getenv('CHAT_OPENAI_API_VERSION'),
            openai_api_type=os.getenv('CHAT_OPEN_API_TYPE'),
            max_tokens=1024
            )
        self.visual_clue_tree_path = visual_clue_tree_path
        
    def get_visual_parts(self):
        visual_parts_prompt = """
        I would like to get a dictionary (json format) of the main visual parts of a given domain, themselves subdivided in visual parts, and so on.
        A visual part is something concrete, and must not be a characteristic or attribute (size, shape, lenght, colour, texture, luminosity).
        The output should be a well-formatted JSON instance that conforms to the JSON schema below. Avoid spaces and '\n' in the output
        As an exemple for the schema: {{'visual_part_1': {{'visual_part_1.1' :{{}}}}, {{'visual_part_1.2': {{}}}}, 'visual_part_2': {{'visual_part_2.1':{{}}, 'visual_part_2.2' : {{'visual_part_2.2.1':{{}}, 'visual_part_2.2.2':{{}}}}, 'visual_part_2.3':{{}}}}, 'visual_part_3': {{}}}}
        If there is no more subdivision, the value of a key-value pair should be an empty Python dictionary.
        One exemple : if the domain is "plane", the schema should be : {{'fuselage':{{'cockpit':{{}}, 'corgo_holds':{{}}, 'passenger_seats':{{}}}}, 'wings':{{'ailerons':{{}}, 'trailing_edge':{{}}, 'leading_edge':{{}}}}, 'landing_gear':{{'wheels':'tyres':{{}}, 'rims':{{}}}}, 'brakes':{{}}}}, 'engine':{{'turbines':{{}}, 'compressors':{{}}}}}}
        The visual parts (and the subdivisions) should be precise, but not too much. Some limitations :
        - there should be between 5 and 10 visual_parts at the first level of the tree
        - then between 0 and 5 at the second level
        - then between 0 and 5 at the third level
        - there should not be more than 3 levels
        Avoid spaces and '\n' in the output, pure JSON format.
        The domain for which I need the dictionary is : {}
        """
        return visual_parts_prompt.format(self.class_domain)

    def get_visual_attributes(self, visual_parts_dict:dict):
        visual_attributes_prompt = """
        I would like to get the visual attributes of things.
        Visual attriutes could be : size, lenght, width, shape, color, texture, luminosity, opacity, material, thickness, level or moisture, level of flexibility. Do not limit to this list, there could be far more relevant visual attributes according to the visual parts.
        It is important to be precise and not to generic.
        Some examples : 
        - the visual attributes for a flower petals are : 'size', 'shape', 'color', 'peduncle size', 'incurvation'
        - the visual attributes for a motor engine : 'speed', 'number of turbines', 'position in the car'
        - the visual attributes for teeth : 'size', 'color', 'sharpness', 'alignment', 'healthy level'
        Fill in each empty dictionary of the input dictionary below by the corresponding visual attributes of the dictionary key, and add en empty dictionary as visual attribute value.
        The output should be a well-formatted JSON instance that conforms to the JSON schema below. Avoid spaces and '\n' in the output.
        Format example : 
        - if the input is : {{'visual_part_1': {{'visual_part_1.1' :{{}}}}, {{'visual_part_1.2': {{}}}}, 'visual_part_2': {{'visual_part_2.1':{{}}, 'visual_part_2.2' : {{'visual_part_2.2.1':{{}}, 'visual_part_2.2.2':{{}}}}, 'visual_part_2.3':{{}}}}}}
        - the output should be : {{'visual_part_1': {{'visual_part_1.1' :{{visual_attributes_1.1.1 : {{}}, visual_attributes_1.1.2 : {{}}}}}}, {{'visual_part_1.2': {{visual_attributes_1.2.1 : {{}}, visual_attributes_1.2.2 : {{}}, , visual_attributes_1.2.3 : {{}}}}}}}}
        There should be between 3 and 7 visual attribute for each visual part.
        The input dictionary is : {}
        """
        return visual_attributes_prompt.format(visual_parts_dict)
       
    def get_attribute_values(self, class_label:str, visual_attributes_dict:dict):
        attribute_values_prompt = """
        For a given subclass (e.g. dog breed), I would like to get the attribute value of each visual attribute (e.g. the size, the shape, the color...) of each visual part (e.g. head, body, legs...), taking into account the global class (e.g. dog).
        Some exemples : 
        for the global class 'dog', the subclass 'rottweiler', the visual part 'head', the sub_visual_part 'teeth', the values are :
        - for the visual attribute 'size' : ' quite large'
        - for the visual attribute 'color' : 'white'
        - for the visual attribute 'sharpness' : 'very sharp'
        - for the visual attribute 'alignement' : 'scissor bite'
        The value should consider how the subclass is postionned in the whole global_class ('rottweiler' are, among the 'dogs', those who have 'very sharp' 'teeth')
        Sometimes, try to be more precise than the split "low" / "medium" / "high" if you can. For instance, you can specify measurement intervals : e.g. "less than 5 inches" / "5-10 inches" / "more than 10 inches" (of course it depends on the unit of measurement)
        Replace each empty dictionary of the input dictionary below by the corresponding attribute value of the dictionary key.
        The output should be a well-formatted JSON instance that conforms to the JSON schema below. Avoid spaces and '\n' in the output.
        Format example :
        - if the input is :{{'visual_part_1': {{'visual_part_1.1' :{{visual_attributes_1.1.1 : {{}}, visual_attributes_1.1.2 : {{}}}}}}, {{'visual_part_1.2': {{visual_attributes_1.2.1 : {{}}, visual_attributes_1.2.2 : {{}}, , visual_attributes_1.2.3 : {{}}}}}}}}
        - the output should be : {{'visual_part_1': {{'visual_part_1.1' :{{visual_attributes_1.1.1 : attribute_value_1.1.1, visual_attributes_1.1.2 : attribute_value_1.1.2}}}}, {{'visual_part_1.2': {{visual_attributes_1.2.1 : attribute_value_1.2.1, visual_attributes_1.2.2 : attribute_value_1.2.2, , visual_attributes_1.2.3 : attribute_value_1.2.3}}}}}}
        The global class is : {}
        The subclass is : {}
        The input dictionary is : {}
        """
        return attribute_values_prompt.format(self.class_domain, class_label, visual_attributes_dict)
    
    def get_visual_clues_natural_language(self, visual_dict:dict):
        visual_clues_sentences_prompt = """
        The objective is to transform a nested Python dictionary values in natural language sentences by aggregatings all information of the nested brand in the sentence.
        The output should be a well-formatted JSON instance, the same as the input, but with the values replaced by string sentences. Avoid spaces and '\n' in the output.
        Format example :
        - if the input is : {{'visual_part_1': {{'visual_part_1.1' :{{visual_attributes_1.1.1 : attribute_value_1.1.1, visual_attributes_1.1.2 : attribute_value_1.1.2}}}}, {{'visual_part_1.2': {{visual_attributes_1.2.1 : attribute_value_1.2.1, visual_attributes_1.2.2 : attribute_value_1.2.2, , visual_attributes_1.2.3 : attribute_value_1.2.3}}}}}}
        - the output should be like : {{'visual_part_1': {{'visual_part_1.1' :{{visual_attributes_1.1.1 : 'the visual_attributes_1.1.1 of the visual_part_1.1 is attribute_value_1.1.1', visual_attributes_1.1.2 : 'the visual_attributes_1.1.2 of the visual_part_1.1 is attribute_value_1.1.2'}}}}, {{'visual_part_1.2': {{visual_attributes_1.2.1 : 'the visual_attributes_1.2.1 of the visual_part_1.2 is attribute_value_1.2.1', visual_attributes_1.2.2 : 'the visual_attributes_1.2.2 of the visual_part_1.2 is attribute_value_1.2.2', , visual_attributes_1.2.3 : 'the visual_attributes_1.2.3 of the visual_part_1.2 is attribute_value_1.2.3'}}}}}}
        Make sure that sentences created are coherent.
        The input dictionary is : {}
        """
        return visual_clues_sentences_prompt.format(visual_dict)
        
    def get_visual_clues_class_domain_dict(self):
        
        # 
        if os.path.exists(self.visual_clue_tree_path):
            print(f"The file '{self.visual_clue_tree_path}' already exists.")
            with open(self.visual_clue_tree_path, "r") as f:
                self.class_domain_dict = json.load(f)
        else:
        
            # get visual parts dict of the class_domain
            visual_parts_dict = self.llm_model.predict(self.get_visual_parts())
            
            # enrich dict with all visual attributes for each visual part of the class_domain
            visual_parts_attributes_dict = self.llm_model.predict(self.get_visual_attributes(visual_parts_dict))
            clean_visual_parts_attributes_dict = re.sub(r' {2,}', '', visual_parts_attributes_dict).replace("\n", "")
            
            # now that we get the tree structure for the class_domain, let's have a dedicated dictionary for each class_label
            # we needed to ensure that visual_parts x visual_attributes are the same for all class_label
            self.class_domain_dict = {}
            
            for class_label in self.class_labels_list:
            
                # enrich dict with attribute values of the class_label for each visual attribute of the class_domain
                visual_parts_attributes_values_dict = self.llm_model.predict(self.get_attribute_values(clean_visual_parts_attributes_dict, class_label))
                self.clean_dict_str = re.sub(r' {2,}', '', visual_parts_attributes_values_dict).replace("\n", "")
                
                # transform attributes values by a natural language sentence that gathers all information of the visual clue (the tree branch)
                dict_with_sentences = self.llm_model.predict(self.get_visual_clues_natural_language(self.clean_dict_str))
                clean_dict_with_sentences = re.sub(r' {2,}', ' ', dict_with_sentences).replace("\n", "")
                self.clean_dict_with_sentences = ast.literal_eval(clean_dict_with_sentences)
                
                # flatten the dictionary by aggregating all branch information in one single key
                flatten_dict_class_label = flatten_dict(self.clean_dict_with_sentences)
                
                # add the class_label dict to the class_domain dict
                self.class_domain_dict[class_label] = flatten_dict_class_label
                
                if self.visual_clue_tree_path:
                    with open(f"{self.visual_clue_tree_path}.json", "w") as f:
                        json.dump(self.class_domain_dict, f)
            
        # rearrange visual clues
        self.sorted_values_dict = {}
        
        for sub_dict in self.class_domain_dict.values():
            for sub_key, value in sub_dict.items():
                if sub_key not in self.sorted_values_dict:
                    self.sorted_values_dict[sub_key] = []
                self.sorted_values_dict[sub_key].append(value)       
        self.sorted_values_dict = {key:list(set(value)) for key,value in self.sorted_values_dict.items()}
        
        self.class_domain_visual_clues_list = [value for values in self.sorted_values_dict.values() for value in values]
        self.class_domain_visual_clues_list = list(set(self.class_domain_visual_clues_list))

        
    def get_visual_clues_embeddings(self, visual_tree_path=None):

        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        
        self.visual_clues_embeddings = {}
        
        for visual_clue in self.class_domain_visual_clues_list:
            tokens = open_clip.tokenize([visual_clue])
            with torch.no_grad():
                text_embedding = clip_model.encode_text(tokens)
            self.visual_clues_embeddings[visual_clue] = text_embedding
            
    def get_class_label_tree_image(self, class_label, image_path):
        
        # Create a new pydot graph
        graph = pydot.Dot(graph_type='digraph')
        
        # Add nodes and edges recursively from the nested dictionary
        graph.add_node(pydot.Node(class_label))
        dict_to_pydot(graph, self.class_domain_dict[class_label], class_label)
        
        # Save the graph to a PNG file and display it
        graph.write_png(f'{image_path}.png')
        display(Image(filename=f'{image_path}.png'))
    