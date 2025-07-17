#language identifier
from huggingface_hub import hf_hub_download
import fasttext

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")   
model_glotlid = fasttext.load_model(model_path)

# ISO 639-2 to 639-1 lang code mapping
GLOT_LANG_DICT = {
    'bul': 'bg',  # Bulgarian
    'ces': 'cs',  # Czech
    'dan': 'da',  # Danish
    'deu': 'de',  # German
    'ell': 'el',  # Greek
    'eng': 'en',  # English
    'spa': 'es',  # Spanish
    'est': 'et',  # Estonian
    'ekk': 'et',  # Standard Estonian
    'fin': 'fi',  # Finnish
    'fra': 'fr',  # French
    'gle': 'ga',  # Irish
    'hrv': 'hr',  # Croatian
    'hun': 'hu',  # Hungarian
    'ita': 'it',  # Italian
    'lit': 'lt',  # Lithuanian
    'lav': 'lv',  # Latvian
    'lvs': 'lv',  # Standard Latvian
    'mlt': 'mt',  # Maltese
    'nld': 'nl',  # Dutch
    'pol': 'pl',  # Polish
    'por': 'pt',  # Portuguese
    'ron': 'ro',  # Romanian
    'slk': 'sk',  # Slovak
    'slv': 'sl',  # Slovenian
    'swe': 'sv'   # Swedish
}

def detect_language(text):
    """Given a text, it returns the Glotlid prediction as NLLB language code, e.g., Latn-eng
    """
    lang_code, score = model_glotlid.predict(text)
    # extract 639-2 lang code (three-letter code)
    three_lang_code = lang_code[0].replace("__label__","").replace("_Latn","")
    # map to 639-1 code if available
    two_letter_code = GLOT_LANG_DICT.get(three_lang_code)
    return three_lang_code, two_letter_code