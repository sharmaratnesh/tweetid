#pseudo code for cohen kappa scoring
from sklearn.metrics import cohen_kappa_score

# Define AI-generated codes
ai_codes = [
    "Community Engagement",
    "Leadership Presence",
    "Event Participation",
    "Climate Technology",
    "Innovation & Tech",
    "Investment Focus",
    "Regional Relevance"
]

# Define manually coded themes
manual_codes = [
    "Community event",
    "Speaker",
    "Ocean summit",
    "Climate change",
    "Technology",
    "Investment",
    "Location specific"
]

# Create a mapping dictionary to align both sets to common themes
theme_mapping = {
    "Community Engagement": "Community",
    "Community event": "Community",
    "Leadership Presence": "Speaker",
    "Speaker": "Speaker",
    "Event Participation": "Event",
    "Ocean summit": "Event",
    "Workshop Attendance": "Event",
    "Climate Technology": "Climate",
    "Climate change": "Climate",
    "Innovation & Tech": "Technology",
    "Technology": "Technology",
    "Investment Focus": "Investment",
    "Investment": "Investment",
    "Regional Relevance": "Location",
    "Location specific": "Location",
    "Gratitude": "Gratitude",
    "Being thankful": "Gratitude"
}

# Map both lists to unified themes
ai_mapped = [theme_mapping[code] for code in ai_codes]
manual_mapped = [theme_mapping[code] for code in manual_codes]

# Calculate Cohen's Kappa score
kappa_score = cohen_kappa_score(ai_mapped, manual_mapped)

print("AI Mapped Codes:", ai_mapped)
print("Manual Mapped Codes:", manual_mapped)
print(f"Cohen's Kappa Score: {kappa_score:.2f}")
