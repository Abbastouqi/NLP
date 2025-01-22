from sklearn.metrics import precision_recall_fscore_support

def evaluate_keywords(predicted_keywords, actual_keywords):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_keywords, 
        predicted_keywords, 
        average='weighted'
    )
    return {'precision': precision, 'recall': recall, 'f1': f1}