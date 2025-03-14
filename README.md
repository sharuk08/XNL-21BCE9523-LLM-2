# T5 Sentiment Analysis Project

## Overview
This project fine-tunes a **T5 model** for sentiment analysis on text data. The pipeline includes **data preprocessing, model training, evaluation, and deployment** as a REST API using Flask.

## Project Structure
```
project/
├── README.md              # Documentation about the project
├── requirements.txt       # Python dependencies for pip installation
├── environment.yml        # Conda environment dependencies
├── data/                  # Data directory
│   ├── download_full.py   # Script for downloading dataset
│   ├── preprocess_data.py # Script for data preprocessing
│   ├── raw.csv            # Raw dataset
│   ├── raw_train.csv      # Training dataset
│   ├── raw_test.csv       # Test dataset
├── models/                # Model storage
│   ├── pretrained/        # Pretrained T5 model
│   ├── final_model/       # Fine-tuned sentiment model
├── src/                   # Source code
│   ├── train.py           # Training script
│   ├── evaluate.py        # Model evaluation script
│   ├── inference.py       # Script for making predictions
│   ├── deployment/        # Deployment-related files
│   │   ├── Dockerfile     # Docker configuration for deployment
├── logs/                  # Logs for training, evaluation, or debugging
├── ds_config.json         # Configuration file for DeepSpeed or distributed training
```

## Setup
### **1. Install Dependencies**
You can install dependencies using either pip or conda:
#### **Using pip:**
```bash
pip install -r requirements.txt
```
#### **Using Conda:**
```bash
conda env create -f environment.yml
conda activate llm-finetuning
```

### **2. Download and Preprocess Data**
Run the following scripts to download and preprocess the dataset:
```bash
python data/download_full.py
python data/preprocess_data.py
```

### **3. Train the Model**
To train the T5 model, run:
```bash
python src/train.py
```

### **4. Run Inference**
Once trained, you can test predictions using:
```bash
python src/inference.py
```

### **5. Deploy as an API**
To serve predictions via a Flask API:
```bash
python src/inference.py
```
It runs on **port 5000**, and you can send a POST request to:
```
http://localhost:5000/predict
```
Example payload:
```json
{
  "text": "I love this product!"
}
```

### **6. Docker Deployment**
To deploy with Docker:
```bash
docker build -t sentiment-analysis .
docker run -p 5000:5000 sentiment-analysis
```

## Model Details
- **Base Model:** T5 (Google's Text-to-Text Transfer Transformer)
- **Fine-Tuning:** Done using `datasets` and `transformers` from Hugging Face
- **Optimizer:** AdamW with learning rate scheduling
- **Training Logs:** Saved in the `logs/` folder

## Acknowledgments
This project leverages:
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK](https://www.nltk.org/)
- [Flask](https://flask.palletsprojects.com/)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## License
This project is open-source and available under the MIT License.
