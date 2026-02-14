# STUDENT INFO
NAME: Ayush Wunnava

BITS ID: 2025AA05765

BITS EMAIL: 2025aa05765@wilp.bits-pilani.ac.in

###

## How to Run?
1. pythom -m venv myenv
2. source myenv/Scripts/activate
3. pip install -r requirements.txt

## FLOW
Full Execution Flow of Your Project

There are 3 layers:
   
Training

Evaluation

App (Inference)

### Step 1 — Train Models (Run Once)
  
    - From project root:
  
    - python model/train_models.py
  
     What happens internally:
   
    - Load dataset (dataset/weatherAUS.csv)
   
    - Preprocess data
   
    - Split into train/test
    
    - Fit scaler
    
    - Train all 6 models
    
    Save:

    - Models → trained_models/

    - Scaler → trained_models/

    - Test data → trained_models/

   After this step:
   trained models exist on disk.



## Step 2 — Evaluate Models

    Running:

    - python model/evaluate_models.py

    What happens internally:

    - Load saved test data

    - Load saved scaler

    - Load saved models

    - Generate predictions

    Compute:

        - Accuracy

        - AUC

        - Precision

        - Recall

        - F1

        - MCC

        - Print comparison table




## Project Structure
dataset/           → raw data

model/             → training + evaluation code

trained_models/    → saved artifacts

app.py             → inference

### INFO
.pkl stands for pickle file.
It is a Python file format used to serialize (save) objects to disk.
In my project, .pkl files contain:
- Trained ML models
- Scaler object
- Test dataset


