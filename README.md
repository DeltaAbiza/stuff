# **Transformer Summarization Model**

This project contains a complete pipeline to train a Transformer-based model for text summarization. It includes scripts for tokenization, data loading, and training with PyTorch.

## **Prerequisites**

* Python 3.8 or higher  
* Git LFS (for handling large data files)

## **Setup and Installation**

Follow these steps to set up your local environment and install the necessary dependencies.

### **1\. Create a Virtual Environment**

It is highly recommended to use a virtual environment to manage project-specific dependencies.

From the project's root directory, run:
```bash
python -m venv .venv
```
### **2\. Activate the Virtual Environment**

**On Windows:**
```bash
.\.venv\Scripts\activate
```
**On macOS/Linux:**
```bash
source .venv/bin/activate
```
You will know it's active when you see (.venv) at the beginning of your terminal prompt.

### **3\. Install Dependencies**

Install all the required Python packages from the requirements.txt file.
```bash
pip install -r requirements.txt
```
## **Data Preparation**

The training data is provided in a compressed archive. You must extract it before running the training script.

1. Locate the data.rar file inside the data/ directory.  
2. Extract its contents directly into the data/ folder.

After extraction, your data directory should contain the following files:
```
data/  
├── tokenizer.json  
└── tokenize\_data.jsonl
```
## **Training the Model**

Once the setup and data preparation are complete, you can start the training process.

From the root directory of the project, run the main training script:
```bash
python train.py
```
The script will automatically detect your GPU if available, load the data, and begin training. Checkpoints will be saved periodically in the ./models directory, allowing you to resume training if it's interrupted.