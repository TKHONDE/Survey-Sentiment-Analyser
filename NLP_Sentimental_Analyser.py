import spacy
from textblob import TextBlob
from pandas import read_csv
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load your exported Google Sheet CSV
df = read_csv("wua_responses.csv")

# Function to analyze text
def analyze(text):
    blob = TextBlob(str(text))
    sentiment = blob.sentiment.polarity
    doc = nlp(str(text))
    entities = [ent.text for ent in doc.ents]
    return sentiment, entities

# Identify all text columns automatically
text_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Apply analysis to each text column
for col in text_columns:
    df[f"{col}_Sentiment"], df[f"{col}_Entities"] = zip(*df[col].apply(analyze))

# Summarize sentiment distribution across all text columns
all_sentiments = []
for col in text_columns:
    all_sentiments.extend(df[f"{col}_Sentiment"].tolist())

positive = sum(1 for s in all_sentiments if s > 0)
negative = sum(1 for s in all_sentiments if s < 0)
neutral  = sum(1 for s in all_sentiments if s == 0)

print("Sentiment Summary:")
print(f"Positive: {positive}")
print(f"Negative: {negative}")
print(f"Neutral: {neutral}")

# Plot sentiment distribution
plt.figure(figsize=(6,4))
plt.bar(["Positive", "Negative", "Neutral"], [positive, negative, neutral], color=["green","red","gray"])
plt.title("Sentiment Distribution Across Survey Responses")
plt.ylabel("Number of Responses")
plt.show()

# Summarize most common entities across all text columns
all_entities = []
for col in text_columns:
    for ents in df[f"{col}_Entities"]:
        all_entities.extend(ents)

entity_counts = Counter(all_entities).most_common(10)
print("Top Entities:", entity_counts[:10])

# Generate word cloud of entities
entity_text = " ".join(all_entities)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(entity_text)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Entities in Survey Responses")
plt.show()

# Save enriched dataset with sentiment + entities
df.to_csv("survey_with_sentiment_entities.csv", index=False)
print(df.head())