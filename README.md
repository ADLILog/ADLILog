### ADLILog: Leveraging Log Instructions in Log-based Anomaly Detection

This repositoiry contains the code implementation of ADLILog, a method for log-based anomaly detection in distributed IT systems. 

### To Start:
1. Clone the reposistoriy: git clone https://github.com/ADLILog/ADLILog.git
2. Download the datasets (e.g., HDGS and BGL) from https://github.com/logpai/loghub. Or put your dataset in the data folder. 
3. Install the requirments from requirments.txt
4. Run the model finetuning with adjusting the relative paths to your dataset.

The stored_model folder contains the model learned from the instructions. One can use the adlilog_train_finetune.py to finetune the model for the dataset of interest. The folder auxilairy_labels contains the "abnormal" class from the SL data, alongside other target system labels from SPIRIT and TBIRD. One 
can find these datasets at https://github.com/logpai/loghub. The SL data can be found at https://zenodo.org/record/6376763.
One can use the parser Drain.py to extract the events that are needed for the baseline. Please note that to give the appropriate credits to Pinjia He et al. [Drain: An Online Log Parsing Approach with Fixed Depth Tree](https://github.com/logpai/logparser/tree/master/logparser/Drain).

#### Code Structure

|data
|--- auxiliary_labels
|models
|---implementations
|------ADLILog
|--------- stored_model
|--------- model
|--------- adlilog.py
|--------- adlilog_train_finetune.py
|--------- tokenizer.py
|--------- utils.py
|parsing
