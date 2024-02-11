# py file to run multiple binary classification with Cross-Validation (CV) and Learning Curves (LC) on CEReD dataset downsampled

##### import ##########
import numpy as np
import pandas as pd
import torch
import evaluate
import argparse
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from datasets import load_metric
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
#########################

# functions :
#   - preprocess_data_for_LC
#   - prepare_model_for_LC
#   - train_test
#   - downsample_dataset

def preprocess_data_for_LC(df_sentences_train, df_sentences_test):
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    # Preprocess data and labels
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    max_length_train = max(df_sentences_train['sentence'].apply(lambda sentence: len(sentence.split())))
    max_length_test = max(df_sentences_test['sentence'].apply(lambda sentence: len(sentence.split())))

    df_train['text'] = df_sentences_train['sentence'].apply(
        lambda x: tokenizer.encode(x, add_special_tokens=True, padding='max_length', truncation = True, max_length=max_length_train))
    df_test['text'] = df_sentences_test['sentence'].apply(
        lambda x: tokenizer.encode(x, add_special_tokens=True, padding='max_length', truncation = True, max_length=max_length_test))
    
    label_encoder_train = LabelEncoder()
    label_encoder_test = LabelEncoder()
    df_train['label'] = label_encoder_train.fit_transform(df_sentences_train['y']) 
    df_test['label'] = label_encoder_test.fit_transform(df_sentences_test['y']) # in output for evaluation
    
    
    # Create a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, text, label):
            self.text = text
            self.label = label
        def __len__(self):
            return len(self.text)
        def __getitem__(self, idx):
            return {
                'text': torch.tensor(self.text[idx], dtype=torch.long),
                'label': torch.tensor(self.label[idx], dtype=torch.long)
                }
    train_dataset = CustomDataset(df_train['text'].values, df_train['label'].values)
    test_dataset = CustomDataset(df_test['text'].values, df_test['label'].values)

    return train_dataset, test_dataset, label_encoder_test

def prepare_model_for_LC(model, train_dataset, test_dataset, freeze_weights, batch_size, epochs, learning_rate):

    if freeze_weights:
        # Freeze all layers except the last two
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    return train_loader, test_loader, optimizer, scheduler 


def train_test(model, train_loader, val_loader, epochs, optimizer, scheduler, device):
    accuracy_metric = load_metric("accuracy")
    train_losses = []
    val_losses = []
    avg_train_acc_per_epoch = []
    avg_val_acc_per_epoch = []
    train_confidence_scores_avg_per_epoch = []  # Store confidence scores for train set
    val_confidence_scores = []    # Store confidence scores for validation set
    avg_balanced_train_acc_per_epoch = [] 
    avg_balanced_val_acc_per_epoch = [] 

    for epoch in range(epochs):
    #for epoch in tqdm(range(epochs), desc="Epochs"):
        #print(f"epoch {epoch} running...")
        model.train()
        train_loss = []
        all_preds_train = []
        all_labels_train = []
        all_train_confidence = []

        #for batch in train_loader:
        for batch in tqdm(train_loader, position = 0, desc= f"epoch {epoch} running..."):
            optimizer.zero_grad()
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            train_loss.append(loss.item())
            predictions_train = torch.argmax(outputs.logits, axis=1)
            all_preds_train.extend(predictions_train.cpu().numpy().tolist())
            all_labels_train.extend(labels.tolist())
            ### compute confidence score
            probabilities = torch.softmax(outputs.logits, dim=1)
            all_train_confidence.extend(probabilities.max(dim=1).values.cpu().detach().numpy())  # Confidence scores
            ###
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_epoch_loss_train = sum(train_loss) / len(train_loss)
        train_losses.append(avg_epoch_loss_train)
        avg_train_acc_per_epoch.append(accuracy_metric.compute(predictions=all_preds_train, references=all_labels_train)["accuracy"])
        avg_balanced_train_acc_per_epoch.append(balanced_accuracy_score(all_labels_train, all_preds_train))
        train_confidence_scores_avg_per_epoch.append(np.mean(all_train_confidence))  # Store confidence scores
        
        # Validation loop
        model.eval()
        val_loss = []
        all_preds_val = []
        all_labels_val = []
        val_confidence = []

        with torch.no_grad():
            for batch in val_loader: 
            #for batch in val_loader:
                inputs = batch['text'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs, labels=labels)
                loss_val = outputs.loss
                val_loss.append(loss_val.item())
                predictions_val = torch.argmax(outputs.logits, axis=1)
                all_preds_val.extend(predictions_val.cpu().numpy().tolist())
                all_labels_val.extend(labels.tolist())
                probabilities = torch.softmax(outputs.logits, dim=1)
                val_confidence.extend(probabilities.max(dim=1).values.cpu().detach().numpy())  # Confidence scores

        avg_epoch_loss_val = sum(val_loss) / len(val_loss)
        val_losses.append(avg_epoch_loss_val)
        avg_val_acc_per_epoch.append(accuracy_metric.compute(predictions=all_preds_val, references=all_labels_val)["accuracy"])
        avg_balanced_val_acc_per_epoch.append(balanced_accuracy_score(all_labels_val, all_preds_val))
        val_confidence_scores.append(np.mean(val_confidence))  # Store confidence scores

    return train_losses, val_losses, avg_train_acc_per_epoch, avg_val_acc_per_epoch, train_confidence_scores_avg_per_epoch, val_confidence_scores, all_train_confidence, avg_balanced_train_acc_per_epoch, avg_balanced_val_acc_per_epoch


def downsample_dataset(df, max_values):
    label_counts = dict(df["y"].value_counts().sort_values(ascending=False))

    downsampled_label_counts = label_counts
    for key, value in label_counts.items():
        downsampled_label_counts[key] = value if value < max_values else max_values

    # Downsample to the maximum desired labels per class
    under_sampler = RandomUnderSampler(sampling_strategy= downsampled_label_counts)
    df_downsampled, _ = under_sampler.fit_resample(df, df['y'])

    # sample the dataset randomly
    df_downsampled = df_downsampled.sample(frac = 1)

    return df_downsampled

######################################################################
################### Main function to run the pipeline ################
######################################################################

def main(batch_size, epochs, N_shuffle_total, topN, cut_downsampling):
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and preprocess
    sentences_en_tr = pd.read_csv('./data/sentences/en/train/sentences.tsv',sep='\t')
    sentences_en_val = pd.read_csv('./data/sentences/en/val/sentences.tsv',sep='\t')
    sentences_en_te = pd.read_csv('./data/sentences/en/test/sentences.tsv',sep='\t')
    
    # Change Difficuties to Difficulty in 'y' column of test dataset
    sentences_en_te['y'] = np.where(sentences_en_te['y'] == 'Difficulties', 'Difficulty', sentences_en_te['y'])

    # Merge the DataFrames
    merged_df = pd.concat([sentences_en_tr, sentences_en_val, sentences_en_te], ignore_index=True)
    # Remove the 'Reflection' label
    merged_df = merged_df[merged_df['y'] != 'Reflection']
    # Shuffle the merged DataFrame
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Before downsampling :")
    print(f"Length dataset: {len(merged_df)}")
    label_counts = dict(merged_df["y"].value_counts().sort_values(ascending=False))
    print(label_counts)
    
    # Downsample the dataset
    dataset_downsampled = downsample_dataset(merged_df, cut_downsampling)
    
    print("\nAfter downsampling :")
    print(f"Length downsampled dataset: {len(dataset_downsampled)}")
    label_counts = dict(dataset_downsampled["y"].value_counts().sort_values(ascending=False))
    print(label_counts)
    print(f"Length of train set: {int(len(dataset_downsampled) * 0.8)}")
    print(f"Length of test set: {int(len(dataset_downsampled) * 0.2)}")
    
    
    ######################################
    ############## TRAIN #################
    ######################################
    
    # Select the reflective categories and show their distribution
    reflective_cat_in_order = list(dataset_downsampled["y"].value_counts().sort_values(ascending=False).index)
    reflective_cat_in_order_wo_other = [item for item in reflective_cat_in_order if item != 'Other']
    #print(f"List of reflective categories in order : {reflective_cat_in_order}")

    topN_classes = reflective_cat_in_order_wo_other[:topN]
    reflective_categories = topN_classes
    learning_rate = 2e-5
    freeze_weights = False
    
    training_examples = [500, 1000, 1500, 2000]
    ##############################
    print("\n#############################")
    print("PARAMETERS")
    print("#############################")
    print(f"reflective_categories = {reflective_categories}")
    print(f"topN = {topN}")
    print(f"epochs = {epochs}")
    print(f"N_shuffle_total = {N_shuffle_total}")
    print(f"batch_size = {batch_size}")
    print(f"learning_rate = {learning_rate}")
    print(f"cut_downsampling = {cut_downsampling}")
    print(f"training_examples = {training_examples}")
    ##############################

    # Create a dictionary to store DataFrames
    dfs_train = {}
    dfs_val = {}
    dfs_train_acc = {}
    dfs_train_balanced_acc = {}
    dfs_val_acc = {}
    dfs_val_balanced_acc = {}
    dfs_test_confidence = {}

    print(f"\n\nLaunching Learning curves per class with : {reflective_categories}")

    for cat in reflective_categories:
        print(f"\n\nStarting Learning curves for category : {cat}")
        train_loss_list = []
        train_acc_list = []
        train_balanced_acc_list = []
        val_loss_list = []
        val_acc_list = []
        val_balanced_acc_list = []
        test_confidence_score_list = []
        
        # Prepare dataset for binary classification
        df_sentences_bin = dataset_downsampled.copy()
        df_sentences_bin['y'] = np.where(df_sentences_bin['y'] == cat, cat, 'Other')
        
        for i, N_shuffle in enumerate(range(N_shuffle_total)):
            print(f"\nShuffle {i+1}:")
            # Split between train and test
            train_dataset_bin, test_dataset_bin = train_test_split(df_sentences_bin, test_size=0.2, random_state=42)
            
            # Initialize the pre-trained BERT model for bin clf
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
            
            for nb_train_ex in training_examples:
                print(f"Train with {nb_train_ex} sentences")
                # preprocess sentences 
                train_dataset_bin_pp, test_dataset_bin_pp, label_encoder_test = preprocess_data_for_LC(train_dataset_bin.head(nb_train_ex), test_dataset_bin)
                # prepare model
                train_loader, test_loader, optimizer, scheduler = prepare_model_for_LC(model, train_dataset_bin_pp, test_dataset_bin_pp, freeze_weights, batch_size, epochs, learning_rate)

                train_loss_fold, val_loss_fold, train_acc_fold, val_acc_fold, _, test_confidence_scores, _, train_balanced_acc_fold, val_balanced_acc_fold = train_test(model, train_loader, test_loader, epochs, optimizer, scheduler, device)
                
                train_loss_list.append(train_loss_fold[-1])
                train_acc_list.append(train_acc_fold[-1])
                train_balanced_acc_list.append(train_balanced_acc_fold[-1])
                val_loss_list.append(val_loss_fold[-1])
                val_acc_list.append(val_acc_fold[-1])
                val_balanced_acc_list.append(val_balanced_acc_fold[-1])
                test_confidence_score_list.append(test_confidence_scores[-1])
                
        # Train loss
        train_loss_array = np.array(train_loss_list).reshape(N_shuffle_total, len(training_examples))
        mean_train_loss = np.mean(train_loss_array, axis=0)
        ci_train_loss = np.percentile(train_loss_array, [2.5, 97.5], axis=0)
        
        # Val loss
        val_loss_array = np.array(val_loss_list).reshape(N_shuffle_total,len(training_examples))
        mean_val_loss = np.mean(val_loss_array, axis=0)
        ci_val_loss = np.percentile(val_loss_array, [2.5, 97.5], axis=0)
        
        # Train acc
        train_acc_array = np.array(train_acc_list).reshape(N_shuffle_total,len(training_examples))
        mean_train_acc = np.mean(train_acc_array, axis=0)
        ci_train_acc = np.percentile(train_acc_array, [2.5, 97.5], axis=0)
        
        # Val acc
        val_acc_array = np.array(val_acc_list).reshape(N_shuffle_total,len(training_examples))
        mean_val_acc = np.mean(val_acc_array, axis=0)
        ci_val_acc = np.percentile(val_acc_array, [2.5, 97.5], axis=0)
        
        # Balanced accuracy on train set
        train_balanced_acc_array = np.array(train_balanced_acc_list).reshape(N_shuffle_total,len(training_examples))
        mean_train_balanced_acc = np.mean(train_balanced_acc_array, axis=0)
        ci_train_balanced_acc = np.percentile(train_balanced_acc_array, [2.5, 97.5], axis=0)

        # Balanced accuracy on validation set
        val_balanced_acc_array = np.array(val_balanced_acc_list).reshape(N_shuffle_total,len(training_examples))
        mean_val_balanced_acc = np.mean(val_balanced_acc_array, axis=0)
        ci_val_balanced_acc = np.percentile(val_balanced_acc_array, [2.5, 97.5], axis=0)
        
        # Test Confidence scores
        test_conf_array = np.array(test_confidence_score_list).reshape(N_shuffle_total,len(training_examples))
        mean_test_conf = np.mean(test_conf_array, axis=0)
        ci_test_conf = np.percentile(test_conf_array, [2.5, 97.5], axis=0)

        # Create a DataFrame for Seaborn
        df_train_loss = pd.DataFrame({
            'N_training_examples': np.arange(len(training_examples)),
            'Mean_Train_Loss': mean_train_loss,
            'Lower_CI': ci_train_loss[0],
            'Upper_CI': ci_train_loss[1]})
        
        df_val_loss = pd.DataFrame({
            'N_training_examples': np.arange(len(training_examples)),
            'Mean_Val_Loss': mean_val_loss,
            'Lower_CI': ci_val_loss[0],
            'Upper_CI': ci_val_loss[1]})
        
        df_train_acc = pd.DataFrame({
            'N_training_examples': np.arange(len(training_examples)),
            'Mean_Train_Acc': mean_train_acc,
            'Lower_CI': ci_train_acc[0],
            'Upper_CI': ci_train_acc[1]})
        
        df_val_acc = pd.DataFrame({
            'N_training_examples': np.arange(len(training_examples)),
            'Mean_Val_Acc': mean_val_acc,
            'Lower_CI': ci_val_acc[0],
            'Upper_CI': ci_val_acc[1]})
        
        df_train_balanced_acc = pd.DataFrame({
        'N_training_examples': np.arange(len(training_examples)),
        'Mean_Train_Balanced_Acc': mean_train_balanced_acc,
        'Lower_CI': ci_train_balanced_acc[0],
        'Upper_CI': ci_train_balanced_acc[1]})

        df_val_balanced_acc = pd.DataFrame({
        'N_training_examples': np.arange(len(training_examples)),
        'Mean_Val_Balanced_Acc': mean_val_balanced_acc,
        'Lower_CI': ci_val_balanced_acc[0],
        'Upper_CI': ci_val_balanced_acc[1]})
        
        df_test_confidence = pd.DataFrame({
            'N_training_examples': np.arange(len(training_examples)),
            'Mean_Test_Conf': mean_test_conf,
            'Lower_CI': ci_test_conf[0],
            'Upper_CI': ci_test_conf[1]})
        
        dfs_train[f'{cat}'] = df_train_loss
        dfs_val[f'{cat}'] = df_val_loss
        dfs_train_acc[f'{cat}'] = df_train_acc
        dfs_val_acc[f'{cat}'] = df_val_acc
        dfs_test_confidence[f'{cat}'] = df_test_confidence
        dfs_train_balanced_acc[f'{cat}'] = df_train_balanced_acc 
        dfs_val_balanced_acc[f'{cat}'] = df_val_balanced_acc  

    ##### Print training/validation results #####
    print("\n\n\ Train loss")
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_train[cat]}\n''' ")
    
    print("\n\n\n Val loss")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_val[cat]}\n''' ")
    
    print("\n\n\n Train acc")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_train_acc[cat]}\n''' ")
    
    print("\n\n\n Val acc")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_val_acc[cat]}\n''' ")
    
    print("\n\n\n Train Balanced acc")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_train_balanced_acc[cat]}\n''' ")
    
    print("\n\n\n Val Balanced acc")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_val_balanced_acc[cat]}\n''' ")
    
    print("\n\n\n Test confidence score")
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_test_confidence[cat]}\n''' ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT model on data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--N_shuffle_total', type=int, default=10, help='Number of shuffles')
    parser.add_argument('--topN', type=int, default=7, help='Number of categories to include')
    parser.add_argument('--cut_downsampling', type=int, default=500, help='Maximum number of sentences per class in set')
    args = parser.parse_args()
    main(args.batch_size, args.epochs, args.N_shuffle_total, args.topN, args.cut_downsampling)