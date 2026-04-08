"""src/kg/entity_extractor.py — Medical NER using spaCy + expanded bilingual keyword rules.

Coverage:
  - 50+ diseases  (English + Gujarati)
  - 45+ symptoms  (English + Gujarati)
  - 35+ drugs     (English + Gujarati spellings)
  - 15 treatments
  - 25 body parts  (English + Gujarati)
  - Severity modifiers
"""
from __future__ import annotations
from dataclasses import dataclass, field

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None  # Graceful fallback

DISEASES_MAP: dict[str, str] = {
    # English
    "diabetes": "diabetes", "diabetes mellitus": "diabetes", "type 2 diabetes": "diabetes",
    "hypertension": "hypertension", "high blood pressure": "hypertension", "blood pressure": "hypertension",
    "tuberculosis": "tuberculosis", "tb": "tuberculosis",
    "malaria": "malaria",
    "dengue": "dengue", "dengue fever": "dengue",
    "typhoid": "typhoid", "typhoid fever": "typhoid",
    "hepatitis": "hepatitis", "hepatitis b": "hepatitis", "hepatitis c": "hepatitis",
    "pneumonia": "pneumonia",
    "asthma": "asthma",
    "cancer": "cancer",
    "stroke": "stroke",
    "arthritis": "arthritis", "rheumatoid arthritis": "arthritis",
    "anemia": "anemia", "anaemia": "anemia",
    "cholera": "cholera",
    "high cholesterol": "high cholesterol", "cholesterol": "high cholesterol",
    "obesity": "obesity", "overweight": "obesity",
    "thyroid disorder": "thyroid disorder", "hypothyroidism": "thyroid disorder", "hyperthyroidism": "thyroid disorder",
    "alzheimer's disease": "alzheimer's disease", "alzheimer": "alzheimer's disease",
    "parkinson's disease": "parkinson's disease", "parkinson": "parkinson's disease",
    "epilepsy": "epilepsy", "seizure disorder": "epilepsy",
    "kidney disease": "kidney disease", "chronic kidney disease": "kidney disease", "renal failure": "kidney disease",
    "liver disease": "liver disease", "cirrhosis": "liver disease", "fatty liver": "liver disease",
    "heart disease": "heart disease", "coronary artery disease": "heart disease",
    "heart attack": "heart attack", "myocardial infarction": "heart attack",
    "covid-19": "covid-19", "covid": "covid-19", "coronavirus": "covid-19",
    "influenza": "influenza", "flu": "influenza",
    "chickenpox": "chickenpox",
    "measles": "measles",
    "depression": "depression",
    "anxiety disorder": "anxiety disorder", "anxiety": "anxiety disorder",
    "insomnia": "insomnia",
    "migraine": "migraine",
    "eczema": "eczema",
    "psoriasis": "psoriasis",
    "food poisoning": "food poisoning",
    "jaundice": "jaundice",
    # Gujarati → English
    "ડાયાબિટ": "diabetes", "ડાયાબીટ": "diabetes", "ડાયાબિટીઝ": "diabetes", "ડાયાબીટીઝ": "diabetes", "મધુ પ્રમેહ": "diabetes",
    "ટ્યૂબર્ક્યુલોસિસ": "tuberculosis", "ક્ષય": "tuberculosis", "ક્ષય રોગ": "tuberculosis", "ટીબી": "tuberculosis",
    "કેન્સર": "cancer", "કૅન્સર": "cancer",
    "સ્ટ્રોક": "stroke",
    "અસ્થમા": "asthma", "દમ": "asthma",
    "મેલેરિયા": "malaria",
    "ડેન્ગ્યુ": "dengue",
    "ટાઈફોઈડ": "typhoid", "ટાઇફોઇડ": "typhoid",
    "ન્યુમોનિયા": "pneumonia",
    "હાઈપરટેન્શન": "hypertension", "ઉચ્ચ રક્ત દબાણ": "hypertension", "લોહીનું દબાણ": "hypertension", "બ્લડ પ્રેશર": "hypertension",
    "હૃદય રોગ": "heart disease",
    "હાર્ટ એટેક": "heart attack", "હૃદ્ ઘા": "heart attack",
    "કિડની ની બીમારી": "kidney disease", "મૂત્ર પ્રક્રિયાની સમસ્યા": "kidney disease",
    "કોલેસ્ટ્રોલ": "high cholesterol",
    "સ્થૂળતા": "obesity", "જાડાપણ": "obesity",
    "થાઈરોઈડ": "thyroid disorder",
    "હીપેટાઈટિસ": "hepatitis",
    "કમળો": "jaundice", "ઝાળો": "jaundice",
    "ઈન્ફ્લ્યુએન્ઝા": "influenza", "ફ્લૂ": "influenza",
    "ચેચક": "chickenpox",
    "ઓરી": "measles",
    "ઉદાસી": "depression",
    "ઊંઘ ન આવવી": "insomnia",
    "આધાશીશી": "migraine",
    "ઝેરી ભોજન": "food poisoning",
    "કોવિડ": "covid-19", "કોરોના": "covid-19",
    "એનિમિયા": "anemia", "લોહી ઓછું": "anemia",
    "સંધિવા": "arthritis", "ગઠિયો": "arthritis",
    "અપસ્માર": "epilepsy",
    "હૈજા": "cholera",
}

SYMPTOMS_MAP: dict[str, str] = {
    # English
    "fever": "fever", "high fever": "fever",
    "headache": "headache",
    "cough": "cough", "dry cough": "cough", "wet cough": "cough",
    "fatigue": "fatigue", "tiredness": "fatigue",
    "weakness": "weakness",
    "vomiting": "vomiting", "nausea": "nausea",
    "diarrhea": "diarrhea", "diarrhoea": "diarrhea", "loose motion": "diarrhea",
    "chest pain": "chest pain",
    "shortness of breath": "shortness of breath", "breathlessness": "shortness of breath", "difficulty breathing": "shortness of breath",
    "dizziness": "dizziness", "vertigo": "dizziness",
    "swelling": "swelling", "edema": "swelling",
    "skin rash": "skin rash", "rash": "skin rash",
    "pain": "pain", "body ache": "body ache", "joint pain": "joint pain", "muscle pain": "muscle pain",
    "bleeding": "bleeding", "blood in stool": "blood in stool", "rectal bleeding": "blood in stool",
    "seizure": "seizure", "convulsion": "seizure",
    "confusion": "confusion",
    "blurred vision": "blurred vision", "vision loss": "vision loss",
    "frequent urination": "frequent urination",
    "excessive thirst": "excessive thirst",
    "weight loss": "weight loss", "appetite loss": "appetite loss", "loss of appetite": "appetite loss",
    "abdominal pain": "abdominal pain", "stomach ache": "abdominal pain",
    "back pain": "back pain",
    "itching": "itching",
    "common cold": "common cold", "cold": "common cold", "runny nose": "common cold",
    "sore throat": "sore throat", "toothache": "toothache",
    "chills": "chills", "night sweats": "night sweats",
    "palpitations": "palpitations", "numbness": "numbness",
    # Gujarati → English
    "તાવ": "fever", "ખૂબ તાવ": "fever", "ઊંચો તાવ": "fever",
    "માથાનો દુખાવો": "headache",
    "ઉલ્ટી": "vomiting", "ઊલ્ટી": "vomiting",
    "ઝાડા": "diarrhea", "ઝાડા-ઊલ્ટી": "diarrhea",
    "ખાંસી": "cough", "ઉધરસ": "cough",
    "થાક": "fatigue", "નબળાઈ": "weakness",
    "ચક્કર": "dizziness",
    "ઉબકા": "nausea",
    "છાતીમાં દુખાવો": "chest pain",
    "શ્વાસ લેવામાં તકલીફ": "shortness of breath", "શ્વાસ ફૂલવો": "shortness of breath", "શ્વાસ ચઢવો": "shortness of breath",
    "સોજો": "swelling", "પગ સોજો": "swelling",
    "દુખાવો": "pain", "સાંધાનો દુખાવો": "joint pain", "સ્નાયુનો દુખાવો": "muscle pain", "પીઠ દુખાવો": "back pain", "પેટ દુખાવો": "abdominal pain", "ઉદર પીડા": "abdominal pain",
    "ચામડી ફોલ્લા": "skin rash",
    "લોહી": "bleeding", "ઝાડામાં લોહી": "blood in stool",
    "વાઈ": "seizure",
    "ઝાંખું": "blurred vision", "ઝાંખું દેખાવું": "blurred vision",
    "વારંવાર પેશાબ": "frequent urination",
    "વધુ તરસ": "excessive thirst",
    "વજન ઘટવું": "weight loss",
    "ભૂખ ન લાગવી": "appetite loss",
    "ગળામાં દુખાવો": "sore throat",
    "ઠંડી": "chills", "શરદી": "common cold", "વહેતું નાક": "common cold",
    "ખંજવાળ": "itching", "ઝણઝણાટી": "numbness",
    "હૃદયના ધબકારા વધવા": "palpitations"
}

DRUGS_MAP: dict[str, str] = {
    # generics English
    "paracetamol": "paracetamol", "acetaminophen": "paracetamol",
    "ibuprofen": "ibuprofen", "aspirin": "aspirin",
    "amoxicillin": "amoxicillin", "amoxycillin": "amoxicillin",
    "metformin": "metformin", "insulin": "insulin",
    "atenolol": "atenolol", "amlodipine": "amlodipine",
    "omeprazole": "omeprazole", "pantoprazole": "pantoprazole",
    "metronidazole": "metronidazole", "azithromycin": "azithromycin", "ciprofloxacin": "ciprofloxacin", "doxycycline": "doxycycline",
    "chloroquine": "chloroquine", "artemisinin": "artemisinin", "hydroxychloroquine": "hydroxychloroquine",
    "lisinopril": "lisinopril", "losartan": "losartan",
    "atorvastatin": "atorvastatin", "rosuvastatin": "rosuvastatin",
    "levothyroxine": "levothyroxine",
    "cetirizine": "cetirizine", "loratadine": "loratadine",
    "prednisolone": "prednisolone", "dexamethasone": "dexamethasone",
    "ranitidine": "ranitidine",
    "antacid": "antacid", "antibiotic": "antibiotic", "antibiotics": "antibiotic", "antiviral": "antiviral",
    "vaccine": "vaccine", "vaccination": "vaccine",
    "cough syrup": "cough syrup", "antihistamine": "antihistamine", "diuretic": "diuretic",
    "blood thinner": "anticoagulant", "warfarin": "anticoagulant", "heparin": "anticoagulant",
    # Gujarati → English
    "પેરાસિટામોલ": "paracetamol",
    "ઇબુપ્રોફેન": "ibuprofen",
    "એસ્પિરિન": "aspirin",
    "ઇન્સ્યુલિન": "insulin",
    "મેટફોર્મિન": "metformin",
    "ક્લોરોક્વિન": "chloroquine",
    "એઝિથ્રોમાસીન": "azithromycin",
    "દવા": "medicine", "ગોળી": "tablet", "ઈન્જેક્શન": "injection", "રસી": "vaccine", "કફ સિરપ": "cough syrup"
}

TREATMENTS_MAP: dict[str, str] = {
    "surgery": "surgery", "operation": "surgery",
    "chemotherapy": "chemotherapy", "chemo": "chemotherapy",
    "dialysis": "dialysis",
    "physiotherapy": "physiotherapy", "physical therapy": "physiotherapy",
    "radiation": "radiation therapy", "radiotherapy": "radiation therapy",
    "immunotherapy": "immunotherapy",
    "blood transfusion": "blood transfusion",
    "oxygen therapy": "oxygen therapy",
    "transplant": "organ transplant",
    "bypass": "bypass surgery", "angioplasty": "angioplasty",
    "catheterization": "catheterization", "endoscopy": "endoscopy", "biopsy": "biopsy",
    "rest": "rest", "diet": "dietary management", "dietary": "dietary management",
    "exercise": "exercise therapy", "rehabilitation": "rehabilitation",
    "counseling": "counseling", "psychotherapy": "psychotherapy",
    # Gujarati
    "સર્જરી": "surgery", "શસ્ત્રક્રિયા": "surgery", "ઓપરેશન": "surgery",
    "કીમોથેરાપી": "chemotherapy", "ડાયાલિસિસ": "dialysis",
    "ફિઝિયોથેરાપી": "physiotherapy", "આરામ": "rest",
    "આહાર": "dietary management", "કસરત": "exercise therapy", "ઓક્સિજન": "oxygen therapy"
}

BODY_PARTS_MAP: dict[str, str] = {
    # English
    "heart": "heart", "cardiac": "heart",
    "lung": "lung", "lungs": "lung", "pulmonary": "lung",
    "liver": "liver", "hepatic": "liver",
    "kidney": "kidney", "renal": "kidney", "kidneys": "kidney",
    "brain": "brain", "cerebral": "brain",
    "stomach": "stomach", "gastric": "stomach",
    "pancreas": "pancreas", "pancreatic": "pancreas",
    "intestine": "intestine", "bowel": "intestine", "colon": "colon",
    "spleen": "spleen", "thyroid": "thyroid gland",
    "bone": "bone", "bones": "bone",
    "muscle": "muscle", "nerve": "nerve", "skin": "skin",
    "eye": "eye", "eyes": "eye", "ear": "ear", "ears": "ear", "nose": "nose", "throat": "throat",
    "spine": "spine", "spinal": "spine",
    "blood": "blood", "vein": "vein", "artery": "artery",
    "joint": "joint", "joints": "joint",
    "gall bladder": "gall bladder", "gallbladder": "gall bladder",
    "uterus": "uterus", "prostate": "prostate", "bladder": "bladder",
    # Gujarati
    "હૃદય": "heart", "ફેફસા": "lung", "યકૃત": "liver", "કિડની": "kidney", "મગજ": "brain",
    "પેટ": "stomach", "આંતરડા": "intestine", "હાડકા": "bone", "સ્નાયુ": "muscle",
    "ચામડી": "skin", "આંખ": "eye", "કાન": "ear", "નાક": "nose", "ગળું": "throat",
    "કરોડરજ્જુ": "spine", "લોહી": "blood", "નસ": "vein", "સાંધા": "joint"
}

SEVERITY_MAP: dict[str, str] = {
    "severe": "high_severity", "serious": "high_severity", "critical": "high_severity",
    "emergency": "high_severity", "life-threatening": "high_severity",
    "chronic": "chronic", "acute": "acute",
    "mild": "mild", "moderate": "moderate",
    # Gujarati
    "ગંભીર": "high_severity", "ખતરનાક": "high_severity", "સામાન્ય": "mild", "હળવો": "mild"
}

@dataclass
class ExtractedEntities:
    diseases: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    severity: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "diseases": self.diseases,
            "symptoms": self.symptoms,
            "drugs": self.drugs,
            "treatments": self.treatments,
            "body_parts": self.body_parts,
            "severity": self.severity,
        }

    def has_entities(self) -> bool:
        return any([self.diseases, self.symptoms, self.drugs, self.treatments, self.body_parts])

def _keyword_match(text: str, keyword_map: dict[str, str]) -> list[str]:
    """
    Match keywords in text, longest phrases first (greedy).
    Works for both English and Gujarati.
    Returns deduplicated canonical values only.
    """
    text_lower = text.lower()
    # Sort by key length descending → longest phrase wins
    sorted_kws = sorted(keyword_map.keys(), key=len, reverse=True)
    seen_canonical: set[str] = set()
    matches: list[str] = []
    
    for kw in sorted_kws:
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            canonical = keyword_map[kw]
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                matches.append(canonical)
                
    return matches

def extract_entities(text: str) -> ExtractedEntities:
    """
    Extract medical entities from English or Gujarati text.
    Uses greedy keyword matching, plus spaCy NER for English.
    """
    result = ExtractedEntities(
        diseases=_keyword_match(text, DISEASES_MAP),
        symptoms=_keyword_match(text, SYMPTOMS_MAP),
        drugs=_keyword_match(text, DRUGS_MAP),
        treatments=_keyword_match(text, TREATMENTS_MAP),
        body_parts=_keyword_match(text, BODY_PARTS_MAP),
        severity=_keyword_match(text, SEVERITY_MAP),
    )

    # Augment with spaCy NER (English text only — detect by ASCII ratio)
    if not text:
        return result
        
    ascii_ratio = sum(1 for c in text[:100] if ord(c) < 128) / max(len(text[:100]), 1)
    if _nlp and ascii_ratio > 0.5:
        doc = _nlp(text[:1024])
        for ent in doc.ents:
            ent_lower = ent.text.lower()
            if ent.label_ in ("DISEASE", "CONDITION"):
                if ent_lower not in result.diseases and len(ent_lower) > 3:
                    result.diseases.append(ent_lower)
            elif ent.label_ == "PRODUCT":
                if ent_lower not in result.drugs and len(ent_lower) > 3:
                    result.drugs.append(ent_lower)

    return result
