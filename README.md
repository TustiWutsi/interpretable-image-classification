# interpretable-image-classification
This repo proposes an LLM-based image classification approach that offers model interpretability possibilities, inspired by this research paper : https://arxiv.org/abs/2405.18672

## Steps to Setup the Project

### 1. Install Dependencies

Start by installing the required Python packages using the `requirements.txt` file.

Open a terminal and run the following command:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies for the project.

### 2. Create and Configure the `.env` File

Next, you'll need to create a `.env` file and add the following environment variables. These variables are needed to connect to Azure and use the OpenAI API.

Create a new file named `.env` in the root directory of the project:

```bash
touch .env
```

and add the following lines:

```env
AZURE_OPENAI_ENDPOINT='your_variable_value'
AZURE_OPENAI_API_KEY='your_variable_value'
CHAT_DEPLOYMENT_NAME='your_variable_value'
CHAT_MODEL_NAME='your_variable_value'
CHAT_OPENAI_API_VERSION='your_variable_value'
CHAT_OPEN_API_TYPE='azure'
```

Replace `'your_variable_value'` with the actual values that you have for your Azure OpenAI setup.

### 3. Download an image dataset

You will need an image dataset to train and test the model. The model was developed with the Standford Dogs Dataset. 
To download the Stanford Dogs Dataset, follow these steps:

#### 3.1 Set Up Your Kaggle API Credentials

- Go to [Kaggle's API page](https://www.kaggle.com/docs/api) and follow the instructions to create a Kaggle API token.
- Download the `kaggle.json` file and move it to the `~/.kaggle/` directory. If this directory does not exist, create it.

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 3.2 Download the Dataset Using Kaggle CLI

Once the Kaggle API credentials are set up, open a terminal and run the following command:

```bash
kaggle datasets download -d jessicali9530/stanford-dogs-dataset
```

This command will download the dataset to your current working directory.

#### 3.3 Unzip the Files

After downloading the dataset, unzip the files with the following command:

```bash
unzip stanford-dogs-dataset.zip -d stanford_dogs
```

This will unzip the dataset into a folder named `stanford_dogs`.

### 4. Run the Code in the Notebook

After setting up your environment and downloading the dataset, open the Jupyter notebook `run.ipynb` and run the code cells sequentially. 
