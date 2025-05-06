# from transformers import pipeline

# # Load the summarization pipeline (this uses a pretrained model from Hugging Face)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def summarize_transcription(transcriptions):
#     """
#     Summarizes the full transcription text.
    
#     Parameters:
#     - transcriptions (list): List of dictionaries containing 'transcription' key
    
#     Returns:
#     - summary (str): Summarized version of the combined transcription text
#     """
#     # Combine all transcriptions into a single string
#     full_text = " ".join([item["transcription"] for item in transcriptions])

#     # Hugging Face models may truncate long texts; break into chunks if needed
#     max_chunk_size = 1024
#     chunks = [full_text[i:i+max_chunk_size] for i in range(0, len(full_text), max_chunk_size)]

#     summary = ""
#     for chunk in chunks:
#         result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
#         summary += result[0]["summary_text"] + " "

#     return summary.strip()


from transformers import pipeline, BartTokenizer

# Initialize summarizer and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_transcription(transcriptions, max_tokens=1024, max_length=150):
    """
    Summarizes transcription text using BART, preserving sentence boundaries.
    
    Args:
        transcriptions (list): List of dicts with "transcription" keys.
        max_tokens (int): Max tokens per chunk (default: 1024, BART's limit).
        max_length (int): Max summary length per chunk (default: 150).
    
    Returns:
        str: Combined summary of all chunks.
    """
    full_text = " ".join([item["transcription"] for item in transcriptions])
    tokens = tokenizer.tokenize(full_text)
    
    # Split into token-limited chunks without breaking sentences
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        if current_length >= max_tokens and token.endswith((".", "!", "?")):
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    
    # Summarize each chunk
    summary = ""
    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
            summary += result[0]["summary_text"] + " "
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    
    return summary.strip()