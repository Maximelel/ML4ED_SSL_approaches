# py file to run the code for CEReD classification

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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from datasets import load_metric
from tqdm import tqdm
#########################

# functions :
#   - preprocess_data_for_CV
#   - prepare_model
#   - train_test
#   - cross_validate
#   - prepare_test_dataset_for_binclf
#   - evaluate_bin

def preprocess_data_for_CV(df_sentences_train, df_sentences_test, train_index, val_index):
    
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
    
    # Split the data into training and validation sets with CV splits
    train_data = df_train.iloc[train_index]
    val_data = df_train.iloc[val_index]
    
    test_data = df_test
    print(f"Train data : {len(train_data)}")
    print(f"Val data : {len(val_data)}")
    print(f"Test data : {len(test_data)}")

    
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
    train_dataset = CustomDataset(train_data['text'].values, train_data['label'].values)
    val_dataset = CustomDataset(val_data['text'].values, val_data['label'].values)
    test_dataset = CustomDataset(test_data['text'].values, test_data['label'].values)

    return train_dataset, val_dataset, test_dataset, label_encoder_test

def prepare_model(model, train_dataset, val_dataset, test_dataset, freeze_weights, batch_size, epochs, learning_rate):

    if freeze_weights:
        # Freeze all layers except the last two
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    return train_loader, val_loader, test_loader, optimizer, scheduler


def train_test(model, train_loader, val_loader, epochs, optimizer, scheduler, device):
    accuracy_metric = load_metric("accuracy")
    train_losses = []
    val_losses = []
    avg_train_acc_per_epoch = []
    avg_val_acc_per_epoch = []
    train_confidence_scores = []  # Store confidence scores for train set
    val_confidence_scores = []    # Store confidence scores for validation set

    #for epoch in range(epochs):
    for epoch in range(epochs):
        #print(f"epoch {epoch} running...")
        model.train()
        train_loss = []
        all_preds_train = []
        all_labels_train = []
        train_confidence = []

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
            train_confidence.extend(probabilities.max(dim=1).values.cpu().detach().numpy())  # Confidence scores
            ###
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_epoch_loss_train = sum(train_loss) / len(train_loss)
        train_losses.append(avg_epoch_loss_train)
        avg_train_acc_per_epoch.append(accuracy_metric.compute(predictions=all_preds_train, references=all_labels_train)["accuracy"])
        train_confidence_scores.append(np.mean(train_confidence))  # Store confidence scores
        
        # Validation loop
        model.eval()
        val_loss = []
        all_preds_val = []
        all_labels_val = []
        val_confidence = []

        with torch.no_grad():
            #for batch in tqdm(val_loader, desc="Validation"): 
            for batch in val_loader:
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
        val_confidence_scores.append(np.mean(val_confidence))  # Store confidence scores

    return train_losses, val_losses, avg_train_acc_per_epoch, avg_val_acc_per_epoch, train_confidence_scores, val_confidence_scores

def cross_validate(df_sentences_train, df_sentences_test, freeze_weights, batch_size, epochs, learning_rate, predictions_list, true_labels_list, n_splits, train_loss_list, train_acc_list, val_loss_list, val_acc_list, device):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_val_acc = 0.0
    best_model = None

    for fold, (train_index, val_index) in enumerate(kf.split(df_sentences_train)):
        print(f"Fold {fold + 1}:")

        train_dataset, val_dataset, test_dataset, label_encoder_test = preprocess_data_for_CV(df_sentences_train, df_sentences_test, train_index, val_index)

        # Initialize the pre-trained BERT model
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9).to(device)
        train_loader, val_loader, test_loader, optimizer, scheduler = prepare_model(model, train_dataset, val_dataset, test_dataset, freeze_weights, batch_size, epochs, learning_rate)
        
        train_loss_fold, val_loss_fold, train_acc_fold, val_acc_fold, _, _ = train_test(model, train_loader, val_loader, epochs, optimizer, scheduler, device)
        
        train_loss_list.extend(train_loss_fold)
        train_acc_list.extend(train_acc_fold)
        val_loss_list.extend(val_loss_fold)
        val_acc_list.extend(val_acc_fold) 

        # Evaluate validation accuracy
        val_accuracy = val_acc_fold[-1]  # Assuming val_acc_fold contains accuracy values for each epoch

        # Update best model if current fold's validation accuracy is higher
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = model.state_dict()  # Store the state dict of the best model
        
    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, label_encoder_test, test_loader, best_model


def prepare_test_dataset_for_binclf(df_sentences_test):

    df_test = pd.DataFrame()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_length_test = max(df_sentences_test['sentence'].apply(lambda sentence: len(sentence.split())))
    df_test['text'] = df_sentences_test['sentence'].apply(
            lambda x: tokenizer.encode(x, add_special_tokens=True, padding='max_length', truncation = True, max_length=max_length_test))

    label_encoder_test = LabelEncoder()
    df_test['label'] = label_encoder_test.fit_transform(df_sentences_test['y'])
    #print(f"Test data : {len(df_test)} sentences")

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

    test_dataset = CustomDataset(df_test['text'].values, df_test['label'].values)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader, label_encoder_test

#if freeze_weights:
#    # Freeze all layers except the last two
#    for param in model.parameters():
#        param.requires_grad = False
#    for param in model.classifier.parameters():
#        param.requires_grad = True

#####################################
def evaluate_bin(model, test_loader, label_encoder, device, accuracy_metric):
    model.eval()
    all_preds = []
    all_labels = []
    pred_confidence = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, axis=1)
            all_preds.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.tolist())
            ### compute confidence score
            probabilities = torch.softmax(outputs.logits, dim=1)
            pred_confidence.extend(probabilities.max(dim=1).values.cpu().detach().numpy())  # Confidence scores

    # compute accuracy
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
    # Decode label encodings
    predicted_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)

    # Get unique labels from true and predicted labels and their union for the confusion matrix
    unique_true_labels = set(predicted_labels)
    unique_predicted_labels = set(true_labels)
    unique_labels_union = unique_true_labels.union(unique_predicted_labels)

    class_labels = sorted(unique_labels_union)

    return accuracy, class_labels, predicted_labels, true_labels

##################################################################
##################################################################

def main(batch_size, epochs, n_splits, topN):
    
    ######################################
    ############## TRAIN #################
    ######################################
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your data from data.csv
    df_sentences_train = pd.read_csv('./data/sentences/en/train/sentences.tsv',sep='\t')
    df_sentences_test = pd.read_csv('./data/sentences/en/test/sentences.tsv',sep='\t')

    # remove category 'Reflexion'
    df_sentences_train = df_sentences_train[df_sentences_train['y'] != 'Reflection']
    df_sentences_test = df_sentences_test[df_sentences_test['y'] != 'Reflection']

    # Select the reflective categories and show their distribution
    reflective_cat_in_order = list(df_sentences_train["y"].value_counts().sort_values(ascending=False).index)
    reflective_cat_in_order_wo_other = [item for item in reflective_cat_in_order if item != 'Other']
    print(f"List of reflective categories in order : {reflective_cat_in_order}")

    topN_classes = reflective_cat_in_order_wo_other[:topN]
    reflective_categories = topN_classes

    learning_rate = 2e-5

    dfs_train = {}
    dfs_val = {}
    dfs_val_acc = {}
    best_model = {}
    predictions_list = []
    true_labels_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    print(f"\n\nLaunching {n_splits}-fold CV per class with : {reflective_categories}")

    for i, cat in enumerate(reflective_categories):
        print(f"\n\nCV for {cat}")
        train_loss_list = []
        val_loss_list = []
        val_acc_list = []
        df_sentences_bin = df_sentences_train.copy()
        df_sentences_bin['y'] = np.where(df_sentences_bin['y'] == cat, cat, 'Other')
        
        train_loss_list, val_loss_list, train_acc_list, val_acc_list, label_encoder_test, _, best_model[cat] = cross_validate(df_sentences_train=df_sentences_bin,
                                                                                                    df_sentences_test = df_sentences_test,
                                                                                                    freeze_weights=False, 
                                                                                                    batch_size=batch_size, 
                                                                                                    epochs=epochs, 
                                                                                                    learning_rate=learning_rate,
                                                                                                    predictions_list = predictions_list,
                                                                                                    true_labels_list = true_labels_list,
                                                                                                    n_splits=n_splits,
                                                                                                    train_loss_list = train_loss_list,
                                                                                                    train_acc_list = train_acc_list,
                                                                                                    val_loss_list = val_loss_list,
                                                                                                    val_acc_list = val_acc_list,
                                                                                                    device = device)

        
        train_loss_array = np.array(train_loss_list).reshape(n_splits,epochs)
        mean_train_loss = np.mean(train_loss_array, axis=0)
        ci_train_loss = np.percentile(train_loss_array, [2.5, 97.5], axis=0)

        val_loss_array = np.array(val_loss_list).reshape(n_splits,epochs)
        mean_val_loss = np.mean(val_loss_array, axis=0)
        ci_val_loss = np.percentile(val_loss_array, [2.5, 97.5], axis=0)
        
        # accuracy on validation set
        val_acc_array = np.array(val_acc_list).reshape(n_splits,epochs)
        mean_val_acc = np.mean(val_acc_array, axis=0)
        ci_val_acc = np.percentile(val_acc_array, [2.5, 97.5], axis=0)

        # Create a DataFrame for Seaborn
        df_train_loss = pd.DataFrame({
            'Epochs': np.arange(epochs),
            'Mean Train Loss': mean_train_loss,
            'Lower CI': ci_train_loss[0],
            'Upper CI': ci_train_loss[1]})

        df_val_loss = pd.DataFrame({
            'Epochs': np.arange(epochs),
            'Mean Val Loss': mean_val_loss,
            'Lower CI': ci_val_loss[0],
            'Upper CI': ci_val_loss[1]})
        
        df_val_acc = pd.DataFrame({
            'Epochs': np.arange(epochs),
            'Mean Val Acc': mean_val_acc,
            'Lower CI': ci_val_acc[0],
            'Upper CI': ci_val_acc[1]})
                                            
        dfs_train[f'{cat}'] = df_train_loss
        dfs_val[f'{cat}'] = df_val_loss
        dfs_val_acc[f'{cat}'] = df_val_acc  


    ##### Print training/validation results #####
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_train[cat]}\n''' ")
    
    print("\n\n\n")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_val[cat]}\n''' ")
    
    print("\n\n\n")
    
    for cat in reflective_categories:
        print(f"\n{cat.lower()} = ''' \n{dfs_val_acc[cat]}\n''' ")

    #####################################
    ############## TEST #################
    #####################################
    print("################################")
    print("################################")
    print("######### START TEST ###########")
    print("################################")
    
    # Load test dataset
    df_sentences_test = pd.read_csv('./data/sentences/en/test/sentences.tsv',sep='\t')
    # Remove 'Reflection' from the dataset
    df_sentences_test = df_sentences_test[df_sentences_test['y'] != 'Reflection']
    # Replace "Difficulties" with "Difficulty" like in the train dataset
    df_sentences_test.loc[df_sentences_test['y'] == 'Difficulties', 'y'] = 'Difficulty'
        
    print(f"Length test dataset: {len(df_sentences_test)}")

    # Initialize empty dictionaries to store data for each category
    accuracy_data = {}
    class_labels_data = {}
    predicted_labels_data = {}
    true_labels_data = {}
    pred_confidence_data = {}

    for i, cat in enumerate(reflective_categories):
        print(cat)
        # preprocess the test dataset for each case : each model has been trained for a binary clf
        df_sentences_test_bin = df_sentences_test.copy()
        df_sentences_test_bin.loc[df_sentences_test_bin['y'] != cat, 'y'] = 'Other'
        
        test_loader, label_encoder_test = prepare_test_dataset_for_binclf(df_sentences_test_bin) 
        
        # Initialize the model
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9).to(device)

        # Load the state dictionary of the best model
        model.load_state_dict(best_model[cat])

        # Evaluate the best model
        accuracy, class_labels, predicted_labels, true_labels, pred_confidence = evaluate_bin(model = model,
                                                                                test_loader = test_loader,
                                                                                label_encoder = label_encoder_test,
                                                                                device = device,
                                                                                accuracy_metric = load_metric("accuracy"))
        # Store data for each category in the respective dictionaries
        accuracy_data[cat] = accuracy
        class_labels_data[cat] = class_labels
        predicted_labels_data[cat] = predicted_labels
        true_labels_data[cat] = true_labels
        pred_confidence_data[cat] = pred_confidence

    print(f"\nOverall accuracy of multiple binary clf : {np.mean(list(accuracy_data.values())).round(4)}")
    # Weighted average accuracy
    weights = [len(df_sentences_test[df_sentences_test['y'] == cat]) for cat in reflective_categories]
    # Calculate the weighted average
    weighted_avg = np.round(sum(w * v for w, v in zip(weights, accuracy_data.values())) / sum(weights),4)
    print(f"\nWeights : {weights}")
    print(f"Weighted accuracy of multiple binary clf : {weighted_avg}\n\n")
    
    print(f"accuracies_bin = {list(accuracy_data.values())}")
    print(f"class_labels = {list(class_labels_data.values())}")

    for cat in reflective_categories:
        print(f"\npredicted_labels_{cat.lower()} = {predicted_labels_data[cat]}\n")
        
    for cat in reflective_categories:
        print(f"\ntrue_labels_{cat.lower()} = {true_labels_data[cat]}\n")
        
    for cat in reflective_categories:
        print(f"\npred_confidence_{cat.lower()} = {pred_confidence_data[cat]}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT model on data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds in CV')
    parser.add_argument('--topN', type=int, default=7, help='Number of categories to include')
    args = parser.parse_args()
    main(args.batch_size, args.epochs, args.n_splits, args.topN)
