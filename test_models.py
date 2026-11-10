# Student Name: mdyusuf | USN: 1MS26CI405
# Stage 2: Hugging Face Model Verification

from transformers import pipeline

STUDENT_NAME = "mdyusuf"
STUDENT_USN = "1MS26CI405"

MODEL_A = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

MODEL_B = (
    "siebert/sentiment-roberta-large-english"
)

review = "This movie was absolutely amazing!"

print("====================================================")
print(f"Student: {STUDENT_NAME} | USN: {STUDENT_USN}")
print("Stage 2: Testing Hugging Face Sentiment Models")
print("====================================================")

print("\nLoading Model A (Initial Champion)...")
model_a = pipeline(
    "sentiment-analysis",
    model=MODEL_A,
    device=-1
)

print("Loading Model B (Challenger)...")
model_b = pipeline(
    "sentiment-analysis",
    model=MODEL_B,
    device=-1
)

print(f"\nSample Review: '{review}'")

print("\nTesting Model A (DistilBERT SST-2):")
print(model_a(review))

print("\nTesting Model B (SiEBERT RoBERTa-Large):")
print(model_b(review))
