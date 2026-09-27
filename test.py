from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.summarize import generate_title, generate_title, summmarize
from utils.audio_processor import process_input, chunk_audio
from core.transcriber import transcribe_all
from dotenv import load_dotenv
load_dotenv()

source = "https://www.youtube.com/watch?v=rBlCOLfMYfw&t=2s"


chunks = process_input(source)
transcript = transcribe_all(chunks, translate=True)

print("\n" + "="*60)
print("Transcript:")
print("\n" + "="*60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


title = generate_title(transcript)
summary = summmarize(transcript)

print("\n" + "="*60)
print("Title:")
print("\n" + "="*60)
print(title)

print("\n" + "="*60)
print("Summary:")
print("\n" + "="*60)
print(summary)

action_items = extract_action_items(transcript)
print("\n" + "="*60)
print("Action Items:")  
print("\n" + "="*60)
print(action_items)

key_decisions = extract_key_decisions(transcript)
print("\n" + "="*60)        
print("Key Decisions:")
print("\n" + "="*60)
print(key_decisions)

questions = extract_questions(transcript)
print("\n" + "="*60)   
print("Open Questions:")
print("\n" + "="*60)
print(questions)
# from dotenv import load_dotenv
# import os

# load_dotenv()

# print(repr(os.getenv("MISTRAL_API_KEY")))