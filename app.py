from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from pymongo import MongoClient
import gridfs
from io import BytesIO
from bson.objectid import ObjectId
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# MongoDB connection setup
client = MongoClient("mongodb+srv://sunithadavuluri8:nri2024@cluster0.g90zt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client['sentiment_analysis']
fs = gridfs.GridFS(db)

# Initialize Flask app
app = Flask(__name__)

# Text preprocessing function
def preprocess_text_with_stopwords(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
        return text
    except Exception:
        return text

# Sentiment analysis function
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return 'happy'
        elif polarity == 0:
            return 'neutral'
        else:
            return 'unhappy'
    except Exception:
        return 'neutral'

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)  # Labels: happy, neutral, unhappy

# Define a function for DistilBERT-based sentiment analysis
def analyze_sentiment_distilbert_with_polarity(preprocessed_texts):
    try:
        inputs = tokenizer(preprocessed_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()
        predictions = torch.argmax(logits, dim=1).tolist()
        sentiment_mapping = {0: 'happy', 1: 'neutral', 2: 'unhappy'}
        
        # Add polarity scores (confidence of prediction)
        polarity_scores = [probs[i][pred] for i, pred in enumerate(predictions)]
        mapped_predictions = [sentiment_mapping[pred] for pred in predictions]

        return mapped_predictions, polarity_scores
    except Exception as e:
        print(f"Error in DistilBERT prediction: {e}")
        return ['neutral'] * len(preprocessed_texts), [0.0] * len(preprocessed_texts)  # Default to neutral and zero polarity


# Route for the upload page
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Route to handle file uploads and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return redirect(url_for('upload_page'))

    # Read the uploaded CSV file
    data = pd.read_csv(file)

    # Step 1: Preprocess the text data
    processed_data = data.copy()
    for col in processed_data.columns:
        processed_data[col] = processed_data[col].apply(preprocess_text_with_stopwords)

    # Step 2: Perform sentiment analysis
    sentiment_data = processed_data.copy()
    for col in sentiment_data.columns:
        sentiment_data[col + '_sentiment'] = sentiment_data[col].apply(analyze_sentiment)

    # Step 3: Generate summary statistics
    facility_sentiment_cols = [col for col in sentiment_data.columns if '_sentiment' in col]
    sentiment_summary = sentiment_data[facility_sentiment_cols].apply(pd.Series.value_counts).fillna(0).astype(int)
    sentiment_summary = sentiment_summary.T
    sentiment_summary.columns = ['happy', 'neutral', 'unhappy']
    overall_sentiment_counts = sentiment_summary.sum()

    # Step 4: Generate Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    overall_sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax_bar)
    ax_bar.set_title('Overall Sentiment Distribution (Bar Chart)')
    ax_bar.set_xlabel('Sentiment')
    ax_bar.set_ylabel('Count')
    plt.tight_layout()
    bar_chart = BytesIO()
    fig_bar.savefig(bar_chart, format='png', bbox_inches='tight')
    bar_chart.seek(0)
    bar_chart_id = fs.put(bar_chart, filename='bar_chart.png')
    plt.close(fig_bar)

    # Step 5: Generate Pie Chart
    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    overall_sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'orange', 'red'], ax=ax_pie)
    ax_pie.set_title('Overall Sentiment Distribution (Pie Chart)')
    ax_pie.set_ylabel('')
    plt.tight_layout()
    pie_chart = BytesIO()
    fig_pie.savefig(pie_chart, format='png', bbox_inches='tight')
    pie_chart.seek(0)
    pie_chart_id = fs.put(pie_chart, filename='pie_chart.png')
    plt.close(fig_pie)

    # Step 6: Generate Stacked Bar Chart
    fig_stack, ax_stack = plt.subplots(figsize=(15, 10))
    sentiment_summary.plot(kind='bar', stacked=True, color=['green', 'orange', 'red'], ax=ax_stack)
    ax_stack.set_title('Facility-Wise Sentiment Distribution', fontsize=16)
    ax_stack.set_xlabel('Facilities', fontsize=14)
    ax_stack.set_ylabel('Count', fontsize=14)
    ax_stack.legend(title='Sentiment', fontsize=12)
    plt.tight_layout()
    stacked_bar_chart = BytesIO()
    fig_stack.savefig(stacked_bar_chart, format='png', bbox_inches='tight')
    stacked_bar_chart.seek(0)
    stacked_bar_chart_id = fs.put(stacked_bar_chart, filename='stacked_bar_chart.png')
    plt.close(fig_stack)

    # Step 7: Generate Individual Facility Pie Charts
    pie_chart_ids = {}
    for facility in sentiment_summary.index:
        fig, ax = plt.subplots(figsize=(8, 8))
        sentiment_summary.loc[facility].plot(kind='pie', autopct='%1.1f%%', colors=['green', 'orange', 'red'], ax=ax)
        ax.set_title(f'{facility} Sentiment Distribution', fontsize=14)
        ax.set_ylabel('')
        plt.tight_layout()
        individual_pie_chart = BytesIO()
        fig.savefig(individual_pie_chart, format='png', bbox_inches='tight')
        individual_pie_chart.seek(0)
        chart_id = fs.put(individual_pie_chart, filename=f'{facility}_pie_chart.png')
        pie_chart_ids[facility] = chart_id
        plt.close(fig)

    # Step 8: Train/Test Split and Model Training
    column_name = processed_data.columns[0]
    bow_vectorizer = CountVectorizer(max_features=1000)
    bow_features = bow_vectorizer.fit_transform(processed_data[column_name])

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_features = tfidf_vectorizer.fit_transform(processed_data[column_name])

    y = sentiment_data[column_name + '_sentiment']
    X_train_bow, X_test_bow, y_train, y_test = train_test_split(bow_features, y, test_size=0.2, random_state=42)
    X_train_tfidf, X_test_tfidf, _, _ = train_test_split(tfidf_features, y, test_size=0.2, random_state=42)

    # Train Models
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    svm = SVC(kernel='linear', random_state=42)

    log_reg.fit(X_train_bow, y_train)
    random_forest.fit(X_train_bow, y_train)
    svm.fit(X_train_bow, y_train)

    # Accuracy Comparison
    bow_accuracies = [
        accuracy_score(y_test, log_reg.predict(X_test_bow)),
        accuracy_score(y_test, random_forest.predict(X_test_bow)),
        accuracy_score(y_test, svm.predict(X_test_bow))
    ]

    tfidf_accuracies = [
        accuracy_score(y_test, log_reg.predict(X_test_tfidf)),
        accuracy_score(y_test, random_forest.predict(X_test_tfidf)),
        accuracy_score(y_test, svm.predict(X_test_tfidf))
    ]

    # Save Accuracy Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = ['Logistic Regression', 'Random Forest', 'SVM']
    x = range(len(model_names))
    bar_width = 0.35

    ax.bar(x, bow_accuracies, width=bar_width, label='BoW', color='skyblue')
    ax.bar([i + bar_width for i in x], tfidf_accuracies, width=bar_width, label='TF-IDF', color='orange')
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.legend()
    plt.tight_layout()

    accuracy_chart = BytesIO()
    fig.savefig(accuracy_chart, format='png', bbox_inches='tight')
    accuracy_chart.seek(0)
    accuracy_chart_id = fs.put(accuracy_chart, filename='accuracy_chart.png')
    plt.close(fig)

    # Confusion Matrices
    fig_cm, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (model, name) in enumerate(zip([log_reg, random_forest, svm], model_names)):
        cm = confusion_matrix(y_test, model.predict(X_test_bow), labels=['happy', 'neutral', 'unhappy'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    plt.tight_layout()

    confusion_chart = BytesIO()
    fig_cm.savefig(confusion_chart, format='png', bbox_inches='tight')
    confusion_chart.seek(0)
    confusion_chart_id = fs.put(confusion_chart, filename='confusion_chart.png')
    plt.close(fig_cm)

    # Generate Bag-of-Words and TF-IDF comparison results
    comparison_results = {}
    for column_name in processed_data.columns:
        if processed_data[column_name].dtype == 'object':  # Only process text columns
            try:
                bow_vectorizer = CountVectorizer(max_features=1000)
                bow_features = bow_vectorizer.fit_transform(processed_data[column_name])
                bow_term_frequencies = bow_features.sum(axis=0).A1

                tfidf_vectorizer = TfidfVectorizer(max_features=1000)
                tfidf_features = tfidf_vectorizer.fit_transform(processed_data[column_name])
                tfidf_term_scores = tfidf_features.sum(axis=0).A1

                comparison_df = pd.DataFrame({
                    'Term': bow_vectorizer.get_feature_names_out(),
                    'Frequency_BoW': bow_term_frequencies,
                    'Frequency_TF-IDF': tfidf_term_scores
                }).sort_values(by='Frequency_BoW', ascending=False).head(20)

                comparison_results[column_name] = {
                    'terms': comparison_df['Term'].tolist(),
                    'bow_values': comparison_df['Frequency_BoW'].tolist(),
                    'tfidf_values': comparison_df['Frequency_TF-IDF'].tolist()
                }
            except Exception as e:
                print(f"Error processing column {column_name}: {e}")


    # Use DistilBERT for predictions on preprocessed data
    true_labels = sentiment_data[column_name + '_sentiment'].map({'happy': 0, 'neutral': 1, 'unhappy': 2}).tolist()
    preprocessed_texts = processed_data[column_name].tolist()
    predicted_labels, polarity_scores = analyze_sentiment_distilbert_with_polarity(preprocessed_texts)

    # Convert predictions back to numerical labels for metric evaluation
    predicted_numeric_labels = [0 if x == 'happy' else 1 if x == 'neutral' else 2 for x in predicted_labels]

    # Compute Precision, Recall, and F1 Score for Multiclass Classification
    precision = precision_score(
        true_labels, 
        predicted_numeric_labels, 
        average=None,  # Calculate precision for each class separately
        labels=[0, 1, 2]
    )
    recall = recall_score(
        true_labels, 
        predicted_numeric_labels, 
        average=None,  # Calculate recall for each class separately
        labels=[0, 1, 2]
    )
    f1 = f1_score(
        true_labels, 
        predicted_numeric_labels, 
        average=None,  # Calculate F1 score for each class separately
        labels=[0, 1, 2]
    )

    # Compute overall average for precision, recall, and polarity for pie chart visualization
    average_precision = sum(precision) / len(precision)
    average_recall = sum(recall) / len(recall)
    average_polarity = sum(polarity_scores) / len(polarity_scores)  # Correct polarity averaging

    # Data for the pie chart
    pie_labels = ['Precision', 'Recall', 'Polarity']
    pie_sizes = [average_precision, average_recall, average_polarity]
    pie_colors = ['skyblue', 'orange', 'lightgreen']

    # Precision and Recall Chart Generation
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    # Labels for categories
    x_labels = ['Happy', 'Neutral', 'Unhappy']
    x = range(len(x_labels))
    bar_width = 0.3

    # Dynamic precision and recall values
    ax_pr.bar(x, precision, width=bar_width, label='Precision', color='skyblue')
    ax_pr.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall', color='orange')

    # Annotate F1 Score on the bars
    for i, f1_score_val in enumerate(f1):
        ax_pr.text(i + bar_width / 2, max(precision[i], recall[i]) + 0.05, f"{f1_score_val:.2f}", ha='center', fontsize=10, color='black')

    # Configure chart
    ax_pr.set_xticks([i + bar_width / 2 for i in x])
    ax_pr.set_xticklabels(x_labels)
    ax_pr.set_ylabel('Score')
    ax_pr.set_title('Precision, Recall, and F1 Score by Sentiment')
    ax_pr.legend()
    plt.tight_layout()

    # Save the chart to MongoDB GridFS
    pr_chart = BytesIO()
    fig_pr.savefig(pr_chart, format='png', bbox_inches='tight')
    pr_chart.seek(0)
    pr_chart_id = fs.put(pr_chart, filename='pr_chart_with_f1.png')
    plt.close(fig_pr)


    # Plot the pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    ax_pie.pie(
        pie_sizes, 
        labels=pie_labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=pie_colors
    )
    ax_pie.set_title('Average Precision, Recall, and Polarity')

    # Save the pie chart as a file in MongoDB GridFS
    pie_chart = BytesIO()
    fig_pie.savefig(pie_chart, format='png', bbox_inches='tight')
    pie_chart.seek(0)
    pie_chart_id_B = fs.put(pie_chart, filename='pie_chart_with_polarity.png')
    plt.close(fig_pie)

    # Render Results with F1 Score and Polarity
    f1_scores = {
        'Happy': round(f1[0], 2),
        'Neutral': round(f1[1], 2),
        'Unhappy': round(f1[2], 2)
    }
    avg_polarity_scores = {
        'Happy': round(sum([polarity_scores[i] for i, label in enumerate(predicted_numeric_labels) if label == 0]) / max(1, predicted_numeric_labels.count(0)), 2),
        'Neutral': round(sum([polarity_scores[i] for i, label in enumerate(predicted_numeric_labels) if label == 1]) / max(1, predicted_numeric_labels.count(1)), 2),
        'Unhappy': round(sum([polarity_scores[i] for i, label in enumerate(predicted_numeric_labels) if label == 2]) / max(1, predicted_numeric_labels.count(2)), 2)
    }

    # Render the result template
    return render_template(
        'result.html',
        bar_chart_id=bar_chart_id,
        pie_chart_id=pie_chart_id,
        pie_chart_id_B=pie_chart_id_B,
        stacked_bar_chart_id=stacked_bar_chart_id,
        pie_chart_ids=pie_chart_ids,
        accuracy_chart_id=accuracy_chart_id,
        confusion_chart_id=confusion_chart_id,
        pr_chart_id=pr_chart_id,  # Dynamically generated Precision-Recall Chart ID
        f1_scores=f1_scores,
        avg_polarity_scores=avg_polarity_scores,
        comparison_results=comparison_results
    )

# Route to serve files from MongoDB
@app.route('/file/<file_id>')
def serve_file(file_id):
    try:
        file = fs.get(ObjectId(file_id))
        return send_file(BytesIO(file.read()), mimetype='image/png')
    except Exception:
        return "File not found", 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
