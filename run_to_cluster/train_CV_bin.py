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

# functions :
#   - preprocess_data_for_CV
#   - prepare_model
#   - train_test
#   - crossvalidate

def preprocess_data_for_CV(df_sentences, train_index, val_index):
    data = pd.DataFrame()
    data['text'] = df_sentences['sentence']
    data['label'] = df_sentences['y']

    # Preprocess data and labels
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    max_length = max(data['text'].apply(lambda sentence: len(sentence.split())))

    data['text'] = data['text'].apply(
        lambda x: tokenizer.encode(x, add_special_tokens=True, padding='max_length', truncation = True, max_length=max_length))

    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    # Split the data into training and validation sets 
    train_data = data.iloc[train_index]
    val_data = data.iloc[val_index]

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

    return train_dataset, val_dataset, label_encoder

def prepare_model(model, train_dataset, val_dataset, freeze_weights, batch_size, epochs, learning_rate):

    if freeze_weights:
        # Freeze all layers except the last two
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    return train_loader, val_loader, optimizer, scheduler #, test_loader

def train_test(model, train_loader, val_loader, epochs, optimizer, scheduler, device):
    accuracy_metric = load_metric("accuracy")
    train_losses = []
    val_losses = []
    avg_train_acc_per_epoch = []
    avg_val_acc_per_epoch = []
    train_confidence_scores = []  # Store confidence scores for train set
    val_confidence_scores = []    # Store confidence scores for validation set

    for epoch in range(epochs):
        print(f"epoch {epoch} running...")
        model.train()
        train_loss = []
        all_preds_train = []
        all_labels_train = []
        train_confidence = []

        for batch in train_loader:
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


def cross_validate(df_sentences, freeze_weights, batch_size, epochs, learning_rate, predictions_list, true_labels_list, n_splits, train_loss_list, train_acc_list, val_loss_list, val_acc_list, device):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(df_sentences)):
        print(f"Fold {fold + 1}:")

        train_dataset, val_dataset, label_encoder = preprocess_data_for_CV(df_sentences, train_index, val_index)
        # Initialize the pre-trained BERT model
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9).to(device)
        train_loader, val_loader, optimizer, scheduler = prepare_model(model, train_dataset, val_dataset, freeze_weights, batch_size, epochs, learning_rate)
        
        train_loss_fold, val_loss_fold, train_acc_fold, val_acc_fold, _, _ = train_test(model, train_loader, val_loader, epochs, optimizer, scheduler, device)
        
        train_loss_list.extend(train_loss_fold)
        train_acc_list.extend(train_acc_fold)
        val_loss_list.extend(val_loss_fold)
        val_acc_list.extend(val_acc_fold)  

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

######################################################################
################### Main function to run the pipeline ################
######################################################################

def main(batch_size, epochs, topN, n_splits):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load your data from data.csv [train/val/test]
    df_sentences = pd.read_csv(f'./data/sentences/en/train/sentences.tsv', sep='\t')

    # Select the reflective categories and show their distribution
    reflective_cat_in_order = list(df_sentences["y"].value_counts().sort_values(ascending=False).index)
    reflective_cat_in_order_wo_other = [item for item in reflective_cat_in_order if item != 'Other']
    #print(f"List of reflective categories in order : {reflective_cat_in_order}")
    #print(f"List of reflective categories in order without 'Other': {reflective_cat_in_order_wo_other}")

    reflective_categories = reflective_cat_in_order_wo_other
    topN_classes = reflective_cat_in_order_wo_other[:topN]
    reflective_categories = topN_classes

    # Set hyperparameters
    learning_rate = 2e-5

    #dfs_train = {}
    #dfs_val = {}
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
        df_sentences_bin = df_sentences.copy()
        df_sentences_bin['y'] = np.where(df_sentences_bin['y'] == cat, cat, 'Other')
        
        train_loss_list, val_loss_list, train_acc_list, val_acc_list = cross_validate(df_sentences=df_sentences_bin,
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
        
        
        print(f"\n\n Category {cat}")
        print("train_loss_array:\n", train_loss_array)
        print("mean_train_loss:\n", mean_train_loss)
        print("ci_train_loss:\n", ci_train_loss)

        val_loss_array = np.array(val_loss_list).reshape(n_splits,epochs)
        mean_val_loss = np.mean(val_loss_array, axis=0)
        ci_val_loss = np.percentile(val_loss_array, [2.5, 97.5], axis=0)
        
        print("val_loss_array:\n", val_loss_array)
        print("mean_val_loss:\n", mean_val_loss)
        print("ci_val_loss:\n", ci_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT model on data')
    #parser.add_argument('--train_file', type=str, help='File name to train data (can be test/train/val)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--topN', type=int, default=3, help='Top N classes')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds in CV')
    args = parser.parse_args()
    main(args.batch_size, args.epochs, args.topN, args.n_splits)