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

LANG_MAP = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
    'cs': 'Czech',
    'pl': 'Polish',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'hr': 'Croatian',
    'bg': 'Bulgarian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mt': 'Maltese',
    'ga': 'Irish'
}

def get_language_name(lang_code, default=None):
    """
    Convert a language code (2 or 3 letters) to full language name.
    
    Args:
        lang_code (str): Two-letter (ISO 639-1) or three-letter (ISO 639-2) language code
        
    Returns:
        str: Full language name, or None if code not found
        
    Examples:
        >>> get_language_name('en')
        'English'
        >>> get_language_name('eng')
        'English'
        >>> get_language_name('de')
        'German'
        >>> get_language_name('deu')
        'German'
    """
    if not lang_code:
        return default
    
    lang_code = lang_code.lower().strip()
    
    # If it's a 2-letter code, try direct lookup in LANG_MAP
    if len(lang_code) == 2:
        return LANG_MAP.get(lang_code)
    
    # If it's a 3-letter code, convert to 2-letter first, then get full name
    elif len(lang_code) == 3:
        two_letter_code = GLOT_LANG_DICT.get(lang_code)
        if two_letter_code:
            return LANG_MAP.get(two_letter_code)
        return default
    
    # Invalid length
    return default


def detect_language(text):
    """Given a text, it returns the Glotlid prediction as NLLB language code, e.g., Latn-eng
    """
    lang_code, score = model_glotlid.predict(text.replace("\n", " "))
    # extract 639-2 lang code (three-letter code)
    three_lang_code = lang_code[0].replace("__label__","").replace("_Latn","")
    # map 639-2 to 639-1 code if available
    two_letter_code = GLOT_LANG_DICT.get(three_lang_code)
    return three_lang_code, two_letter_code