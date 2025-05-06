import os
import sys
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from youtube_audio import download_audio
from transcriber import transcribe_audio
from classifier import classify_sentence, evaluate_classifier
from test_samples import test_samples
from transformers import pipeline, BartTokenizer 

os.environ["SPEECHBRAIN_CACHE_STRATEGY"] = "copy"
load_dotenv()

class BartSummarizer:
    """Handles text summarization using BART model"""
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    def summarize(self, text, max_length=150):
        """
        Summarizes text using BART model with proper chunking
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of summary
            
        Returns:
            str: Generated summary
        """
        # Tokenize and chunk preserving sentence boundaries
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            if current_length >= 1000 and token.endswith((".", "!", "?")):
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
        
        # Summarize each chunk
        summary = ""
        for chunk in chunks:
            try:
                result = self.summarizer(
                    chunk, 
                    max_length=max_length, 
                    min_length=30, 
                    do_sample=False
                )
                summary += result[0]["summary_text"] + " "
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summary += chunk[:max_length//2] + "... "  # Fallback
        
        return summary.strip()

def process_youtube_video(youtube_url):
    """Process a YouTube video through the full pipeline."""
    try:
        print(f"\n{'='*50}\nProcessing YouTube video: {youtube_url}\n{'='*50}")
        
        # 1. Download audio
        print("\n[1/4] Downloading audio...")
        filename = download_audio(youtube_url)
        print(f"✓ Audio saved as: {filename}")
        
        # 2. Transcribe audio
        print("\n[2/4] Transcribing audio...")
        transcription = transcribe_audio(filename)
        print(f"✓ Transcription complete ({len(transcription.split())} words)")
        
        # 3. Classify sentences
        print("\n[3/4] Classifying sentences...")
        sentences = sent_tokenize(transcription)
        
        # Print classification results with confidence
        print("\nClassification Results:")
        classified_sentences = []
        for idx, sent in enumerate(sentences, 1):
            label, confidence = classify_sentence(sent)
            truncated_sent = f"{sent[:60]}..." if len(sent) > 60 else sent
            print(f"{idx:>3}. [{label:<15}] ({confidence:.0%}) {truncated_sent}")
            classified_sentences.append(sent)
        
        # 4. Generate Summary using BART
        print("\n[4/4] Generating summary...")
        summarizer = BartSummarizer()
        summary = summarizer.summarize(transcription)
        
        print(f"\n=== SUMMARY ===\n{summary}\n")
        print(f"Summary length: {len(summary.split())} words (reduced from {len(transcription.split())})")
        
        # # Evaluate classifier (optional)
        # print("\nEvaluating Classifier Performance...")
        # accuracy = evaluate_classifier(test_samples)
        # print(f"✓ Classifier Accuracy: {accuracy:.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n⚠️ Error processing video: {str(e)}")
        return False

def main():
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
        success = process_youtube_video(youtube_url)
        
        if success:
            print("\n✅ Processing completed successfully!")
        else:
            print("\n❌ Processing failed. See error above.")
    else:
        print("Usage: python main.py <YouTube_URL>")
        print("Example: python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    main()