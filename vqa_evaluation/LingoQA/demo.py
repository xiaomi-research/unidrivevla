# Demo for using the Lingo-Judge directly from Hugging Face
from transformers import pipeline

# Define the model name to be used in the pipeline
model_name = 'wayveai/Lingo-Judge'
# Define the question and its corresponding answer and prediction
question = "Are there any pedestrians crossing the road? If yes, how many?"
answer = "1"
prediction = "Yes, there is one"
# Initialize the pipeline with the specified model, device, and other parameters
pipe = pipeline("text-classification", model=model_name)
# Format the input string with the question, answer, and prediction
input = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"
# Pass the input through the pipeline to get the result
result = pipe(input)
# Print the result and score
score = result[0]['score']
print(score > 0.5, score)