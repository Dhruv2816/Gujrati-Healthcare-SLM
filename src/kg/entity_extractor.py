"""src/kg/entity_extractor.py — Medical NER using spaCy + custom Gujarati/English keyword rules."""
from __future__ import annotations
from dataclasses import dataclass, field

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None  # Graceful fallback if spaCy not installed

# Canonical English names used in Neo4j Knowledge Graph
DISEASES_MAP = {
    "diabetes": "diabetes", "hypertension": "hypertension", "tuberculosis": "tuberculosis", "tb": "tuberculosis",
    "malaria": "malaria", "dengue": "dengue", "typhoid": "typhoid", "hepatitis": "hepatitis",
    "pneumonia": "pneumonia", "asthma": "asthma", "cancer": "cancer", "stroke": "stroke",
    "arthritis": "arthritis", "anemia": "anemia", "cholera": "cholera", "cholesterol": "cholesterol",
    "obesity": "obesity", "thyroid": "thyroid", "alzheimer": "alzheimer", "parkinson": "parkinson",
    "epilepsy": "epilepsy", "kidney disease": "kidney disease", "liver disease": "liver disease",
    # Gujarati -> English
    "ડાયાબિટ": "diabetes", "ડાયાબીટ": "diabetes", "ટ્યૂબર્ક્યુલોસિસ": "tuberculosis", 
    "કેન્સર": "cancer", "સ્ટ્રોક": "stroke", "અસ્થમા": "asthma", 
    "મેલેરિયા": "malaria", "ડેન્ગ્યુ": "dengue", "ટાઈફોઈડ": "typhoid", 
    "ન્યુમોનિયા": "pneumonia", "હાઈપરટેન્શન": "hypertension", "લોહીનું દબાણ": "hypertension",
}

SYMPTOMS_MAP = {
    "fever": "fever", "headache": "headache", "cough": "cough", "fatigue": "fatigue",
    "vomiting": "vomiting", "nausea": "nausea", "diarrhea": "diarrhea",
    "chest pain": "chest pain", "shortness of breath": "shortness of breath",
    "dizziness": "dizziness", "weakness": "weakness", "swelling": "swelling",
    "rash": "rash", "pain": "pain", "bleeding": "bleeding", "seizure": "seizure",
    "confusion": "confusion", "blurred vision": "blurred vision",
    # Gujarati -> English
    "તાવ": "fever", "માથાનો દુખાવો": "headache", "ઉલ્ટી": "vomiting", 
    "ઝાડા": "diarrhea", "ખાંસી": "cough", "થાક": "fatigue", "ચક્કર": "dizziness",
    "છાતીમાં દુખાવો": "chest pain", "શ્વાસ લેવામાં તકલીફ": "shortness of breath", 
    "શ્વાસ ફૂલવો": "shortness of breath", "શરદી": "cough", "દુખાવો": "pain",
}

DRUGS_MAP = {
    "paracetamol": "paracetamol", "ibuprofen": "ibuprofen", "aspirin": "aspirin", 
    "amoxicillin": "amoxicillin", "metformin": "metformin",
    "insulin": "insulin", "atenolol": "atenolol", "amlodipine": "amlodipine", 
    "omeprazole": "omeprazole", "antibiotics": "antibiotics",
    "antibiotic": "antibiotics", "antiviral": "antiviral", "vaccine": "vaccine", 
    "metronidazole": "metronidazole", "azithromycin": "azithromycin",
}

TREATMENTS_MAP = {
    "surgery": "surgery", "chemotherapy": "chemotherapy", "dialysis": "dialysis", 
    "physiotherapy": "physiotherapy", "radiation": "radiation",
    "immunotherapy": "immunotherapy", "blood transfusion": "blood transfusion", 
    "oxygen therapy": "oxygen therapy", "transplant": "transplant",
    "bypass": "bypass", "catheterization": "catheterization", "endoscopy": "endoscopy", "biopsy": "biopsy",
}

BODY_PARTS_MAP = {
    "heart": "heart", "lung": "lung", "liver": "liver", "kidney": "kidney", 
    "brain": "brain", "stomach": "stomach", "pancreas": "pancreas",
    "intestine": "intestine", "spleen": "spleen", "thyroid": "thyroid", 
    "bone": "bone", "muscle": "muscle", "nerve": "nerve", "skin": "skin",
    "eye": "eye", "ear": "ear", "nose": "nose", "throat": "throat", "spine": "spine",
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


def _keyword_match(text: str, keyword_map: dict) -> list[str]:
    text_lower = text.lower()
    # Sort by length descending to match longest phrases first (greedy)
    sorted_kws = sorted(keyword_map.keys(), key=len, reverse=True)
    matches = []
    found_text = text_lower
    for kw in sorted_kws:
        if kw.lower() in found_text:
            matches.append(keyword_map[kw])
            # Optional: remove kw from found_text to prevent double counting
            # found_text = found_text.replace(kw.lower(), " ")
    return matches


def extract_entities(text: str) -> ExtractedEntities:
    """
    Extract medical entities from English or Gujarati text.
    Uses spaCy for English NER + keyword matching for both languages.
    """
    result = ExtractedEntities(
        diseases=_keyword_match(text, DISEASES_MAP),
        symptoms=_keyword_match(text, SYMPTOMS_MAP),
        drugs=_keyword_match(text, DRUGS_MAP),
        treatments=_keyword_match(text, TREATMENTS_MAP),
        body_parts=_keyword_match(text, BODY_PARTS_MAP),
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
