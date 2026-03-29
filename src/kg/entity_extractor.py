"""src/kg/entity_extractor.py — Medical NER using spaCy + custom Gujarati/English keyword rules."""
from __future__ import annotations
from dataclasses import dataclass, field

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None  # Graceful fallback if spaCy not installed

# ── Medical keyword dictionaries ─────────────────────────────────────────────
DISEASES = {
    "diabetes", "hypertension", "tuberculosis", "tb", "malaria", "dengue",
    "typhoid", "hepatitis", "pneumonia", "asthma", "cancer", "stroke",
    "arthritis", "anemia", "cholera", "cholesterol", "obesity", "thyroid",
    "alzheimer", "parkinson", "epilepsy", "kidney disease", "liver disease",
    # Gujarati
    "ડાયાબિટ", "ડાયાબીટ", "ટ્યૂબર્ક્યુલોસિસ", "કેન્સર", "સ્ટ્રોક",
}

SYMPTOMS = {
    "fever", "headache", "cough", "fatigue", "vomiting", "nausea", "diarrhea",
    "chest pain", "shortness of breath", "dizziness", "weakness", "swelling",
    "rash", "pain", "bleeding", "seizure", "confusion", "blurred vision",
    # Gujarati
    "તાવ", "માથાનો દુખાવો", "ઉલ્ટી", "ઝાડા", "ખાંસી", "થાક", "ચક્કર",
}

DRUGS = {
    "paracetamol", "ibuprofen", "aspirin", "amoxicillin", "metformin",
    "insulin", "atenolol", "amlodipine", "omeprazole", "antibiotics",
    "antibiotic", "antiviral", "vaccine", "metronidazole", "azithromycin",
}

TREATMENTS = {
    "surgery", "chemotherapy", "dialysis", "physiotherapy", "radiation",
    "immunotherapy", "blood transfusion", "oxygen therapy", "transplant",
    "bypass", "catheterization", "endoscopy", "biopsy",
}

BODY_PARTS = {
    "heart", "lung", "liver", "kidney", "brain", "stomach", "pancreas",
    "intestine", "spleen", "thyroid", "bone", "muscle", "nerve", "skin",
    "eye", "ear", "nose", "throat", "spine",
}


@dataclass
class ExtractedEntities:
    diseases: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diseases": self.diseases,
            "symptoms": self.symptoms,
            "drugs": self.drugs,
            "treatments": self.treatments,
            "body_parts": self.body_parts,
        }

    def has_entities(self) -> bool:
        return any([self.diseases, self.symptoms, self.drugs, self.treatments, self.body_parts])


def _keyword_match(text: str, keyword_set: set) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keyword_set if kw.lower() in text_lower]


def extract_entities(text: str) -> ExtractedEntities:
    """
    Extract medical entities from English or Gujarati text.
    Uses spaCy for English NER + keyword matching for both languages.
    """
    result = ExtractedEntities(
        diseases=_keyword_match(text, DISEASES),
        symptoms=_keyword_match(text, SYMPTOMS),
        drugs=_keyword_match(text, DRUGS),
        treatments=_keyword_match(text, TREATMENTS),
        body_parts=_keyword_match(text, BODY_PARTS),
    )

    # Augment with spaCy NER (English text only)
    if _nlp and any(ord(c) < 128 for c in text[:100]):
        doc = _nlp(text[:512])  # Limit for performance
        for ent in doc.ents:
            if ent.label_ in ("DISEASE", "GPE") and ent.text.lower() not in result.diseases:
                result.diseases.append(ent.text.lower())
            elif ent.label_ == "PRODUCT" and ent.text.lower() not in result.drugs:
                result.drugs.append(ent.text.lower())

    # Deduplicate
    result.diseases  = list(dict.fromkeys(result.diseases))
    result.symptoms  = list(dict.fromkeys(result.symptoms))
    result.drugs     = list(dict.fromkeys(result.drugs))
    result.treatments= list(dict.fromkeys(result.treatments))
    result.body_parts= list(dict.fromkeys(result.body_parts))

    return result
