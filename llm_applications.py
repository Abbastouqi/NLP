from transformers import pipeline

class LLMApplications:
    def __init__(self):
        # Initialize all three pipelines
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.classifier = pipeline("zero-shot-classification")
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def summarize_text(self, text):
        """Text summarization using BART model"""
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    def classify_text(self, text, labels):
        """Text classification with custom labels"""
        result = self.classifier(text, labels)
        return {
            'labels': result['labels'],
            'scores': result['scores']
        }

    def analyze_sentiment(self, text):
        """Sentiment analysis"""
        result = self.sentiment_analyzer(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }

def main():
    # Initialize the LLM applications
    llm_app = LLMApplications()

    # Example text for testing
    text = """
    Artificial intelligence has transformed the way we live and work. 
    From virtual assistants like Siri and Alexa to recommendation systems 
    on streaming platforms, AI is everywhere. Machine learning algorithms 
    power self-driving cars, medical diagnosis systems, and fraud detection 
    in banking. The technology continues to evolve rapidly, with new 
    breakthroughs happening regularly in areas like natural language 
    processing and computer vision.
    """

    # 1. Summarization
    print("\n=== Text Summarization ===")
    print("Original Text:", text)
    summary = llm_app.summarize_text(text)
    print("\nSummary:", summary)

    # 2. Classification
    print("\n=== Text Classification ===")
    labels = ["technology", "science", "business", "entertainment"]
    classification = llm_app.classify_text(text, labels)
    for label, score in zip(classification['labels'], classification['scores']):
        print(f"{label}: {score:.2f}")

    # 3. Sentiment Analysis
    print("\n=== Sentiment Analysis ===")
    sentiment = llm_app.analyze_sentiment(text)
    print(f"Sentiment: {sentiment['label']}")
    print(f"Confidence Score: {sentiment['score']:.2f}")

if __name__ == "__main__":
    main()