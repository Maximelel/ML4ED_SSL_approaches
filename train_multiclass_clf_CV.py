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
#   - evaluate

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


def evaluate(model, test_loader, label_encoder, device, accuracy_metric):
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
            ###

    # compute accuracy
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
    print(f"Accuracy on test dataset: {np.round(accuracy,3)}")
    # Decode label encodings
    predicted_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)

    # Get unique labels from true and predicted labels and their union for the confusion matrix
    unique_true_labels = set(predicted_labels)
    unique_predicted_labels = set(true_labels)
    unique_labels_union = unique_true_labels.union(unique_predicted_labels)

    # Sort the labels alphabetically to ensure consistent order
    class_labels = sorted(unique_labels_union)
    
    #print(f"\n Missing labels : {set(['Belief', 'Difficulty', 'Experience', 'Feeling', 'Other', 'Reflection', 'Learning', 'Perspective', 'Intention']) - unique_labels_union}\n")
    
    return predicted_labels, true_labels, pred_confidence, class_labels

def main(batch_size, epochs, n_splits):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data from train file
    df_sentences_train = pd.read_csv(f'./data/sentences/en/train/sentences.tsv', sep='\t')
    df_sentences_test = pd.read_csv(f'./data/sentences/en/test/sentences.tsv', sep='\t')
    
    # remove category 'Reflexion'
    df_sentences_train = df_sentences_train[df_sentences_train['y'] != 'Reflection']
    df_sentences_test = df_sentences_test[df_sentences_test['y'] != 'Reflection']
    
    learning_rate = 2e-5

    predictions_list = []
    true_labels_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    train_loss_list, val_loss_list, train_acc_list, val_acc_list, label_encoder_test, test_loader, best_model  = cross_validate(df_sentences_train=df_sentences_train,
                                                                                        df_sentences_test=df_sentences_test,
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


    ##### Print training/validation results #####
    print(f"n_splits = {n_splits}")
    print(f"epochs = {epochs}")
    print(f"\ntrain_loss_list = {train_loss_list}")
    print(f"\nval_loss_list = {val_loss_list}")
    print(f"\ntrain_acc_list = {train_acc_list}")
    print(f"\nval_acc_list = {val_acc_list}")

    # Evaluate model on test dataset
    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9).to(device)

    # Load the state dictionary of the best model
    model.load_state_dict(best_model)

    # Evaluate the best model
    predicted_labels, true_labels, pred_confidence, class_labels = evaluate(model = model,
                                                                            test_loader = test_loader,
                                                                            label_encoder = label_encoder_test,
                                                                            device = device,
                                                                            accuracy_metric = load_metric("accuracy"))
    
    ###### Print evaluation results ######
    print(f"\npredicted_labels = {predicted_labels}")
    print(f"\ntrue_labels = {true_labels}")
    print(f"\npred_confidence = {pred_confidence}")
    print(f"\nclass_labels = {class_labels}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT model on data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds in CV')
    args = parser.parse_args()
    main(args.batch_size, args.epochs, args.n_splits)
