"""ocr_patterns.py - Regular-expression patterns for identifying OCR samples.

Based on an analysis of the LLaVA-NeXT-Data 780K dataset, approximately 6.46% of
samples were identified as OCR-related.
"""

import re
from typing import List, Set


# ===== OCR regular-expression patterns =====
# Format: (regular expression, category name)
OCR_PATTERNS = [
    # === 1. Explicit text-reading requests ===
    (r'what (does|is) (the )?text (say|read|written|show)', 'text_query'),
    (r'read (the )?text', 'read_text'),
    (r'reading (the )?(text|word)', 'reading'),
    (r'text (in|on|written on) (the )?(image|sign|paper|screen|picture|poster|board)', 'text_location'),
    (r'what (is |are )?written (on|in)', 'written_query'),
    (r'what (does|do) (the )?(sign|signs|label|labels|banner|board|poster) (say|read|show)', 'sign_text'),
    (r'recognize (the )?(text|character|letter|word|writing)', 'recognize'),
    (r'\bocr\b', 'ocr_keyword'),
    (r'text recognition', 'text_recognition'),
    
    # === 2. License-plate recognition ===
    (r'license plate', 'license_plate'),
    (r'number plate', 'number_plate'),
    (r'plate number', 'plate_number'),
    (r'registration (number|plate)', 'registration'),
    
    # === 3. Number/character recognition ===
    (r'what (number|numbers|digit|digits) (are |is )?(displayed|shown|visible|written|on)', 'number_query'),
    (r'can you (read|see|identify) (the )?(number|digit)', 'number_read'),
    (r'(number|digit) (shown|displayed|visible|written) (in|on)', 'number_location'),
    (r'what (letter|letters|character|characters|alphabet) (are |is )?(shown|displayed|written|on)', 'letter_query'),
    (r'can you (read|identify|recognize) (the )?(letter|character)', 'letter_read'),
    
    # === 4. Transcription/extraction ===
    (r'transcribe', 'transcribe'),
    (r'copy (the )?text', 'copy_text'),
    (r'extract (the )?(text|word|number)', 'extract'),
    
    # === 5. Signage/sign text ===
    (r'what (does|is) (the )?(store|shop|building|street) (sign|name|label) (say|read)', 'store_sign'),
    (r'what (is )?(written|printed) on (the )?(sign|poster|banner|board)', 'sign_written'),
    
    # === 6. Documents/screens ===
    (r'text on (the )?(screen|monitor|display|document|page|book|paper)', 'screen_text'),
    (r'what (is )?(displayed|shown|written) on (the )?(screen|document)', 'screen_display'),
    (r'can you read (the )?(document|paper|page)', 'document_read'),
    
    # === 7. Specific scenarios ===
    (r'what (does|is) (the )?(menu|receipt|ticket|card|label|tag) (say|read|show)', 'menu_text'),
    (r'(menu|receipt|ticket|invoice|form) (say|read|show|contain)', 'menu_content'),
    (r'what (brand|logo|name) (is )?(written|shown|displayed)', 'brand_query'),
    
    # === 8. Handwritten text ===
    (r'handwrit(ten|ing)', 'handwriting'),
    
    # === 9. Other related terms ===
    (r'words (in|on|written|displayed|say|read)', 'words'),
    (r'(signage|signboard|billboard)', 'signage'),
    
    # === 10. Descriptive-text keywords (for descriptive datasets such as Obelics) ===
    # Exclude 'caption': it usually describes the image as metadata rather than text to recognize
    (r'\bwebpage\b', 'webpage'),
    (r'\bwebsite\b', 'website'),
    (r'\bscreenshot\b', 'screenshot'),
    (r'\barticle\b', 'article'),
    (r'\bheadline\b', 'headline'),
    (r'\bnewspaper\b', 'newspaper'),
    (r'\bmagazine\b', 'magazine'),
    (r'\bflyer\b', 'flyer'),
    (r'\bdocument\b', 'document'),
    (r'\bquote\b', 'quote'),
    (r'\bparagraph\b', 'paragraph'),
    (r'\bsentence\b', 'sentence'),
    (r'\btitle\b', 'title'),
    (r'\blogo\b', 'logo'),
    (r'\bpage\b', 'page'),
    (r'\btext\b', 'text'),  # Contains the "text" keyword
    
    # === 11. Document file types ===
    (r'\bpdf\b', 'pdf'),
    (r'\bword\b', 'word'),
    (r'\bdocx?\b', 'doc'),  # doc or docx
    (r'\bpresentation\b', 'presentation'),
    (r'\bspreadsheet\b', 'spreadsheet'),
    (r'\bexcel\b', 'excel'),
    (r'\bpowerpoint\b', 'powerpoint'),
    (r'\bslide(s)?\b', 'slides'),
    (r'\breport\b', 'report'),
    (r'\bform\b', 'form'),
    (r'\binvoice\b', 'invoice'),
    (r'\bcontract\b', 'contract'),
]

# Compile regular expressions for better performance
COMPILED_OCR_PATTERNS = [(re.compile(p, re.IGNORECASE), cat) for p, cat in OCR_PATTERNS]


def is_ocr_sample(messages: List[dict], min_confidence: str = 'low') -> bool:
    """Determine whether a conversation sample is OCR-related.

    Args:
        messages: Message list in the form [{"role": "user", "content": "..."}, ...]
        min_confidence: Minimum confidence: 'high' (strong signals only),
            'medium' (medium signals), or 'low' (all signals)

    Returns:
        bool: Whether this is an OCR sample
    """
    # Combine all message contents
    all_text = ' '.join([msg.get('content', '').lower() for msg in messages])
    
    # Check whether any OCR pattern matches
    matched_categories = set()
    for pattern, category in COMPILED_OCR_PATTERNS:
        if pattern.search(all_text):
            matched_categories.add(category)
    
    if not matched_categories:
        return False
    
    # Filter by confidence
    if min_confidence == 'high':
        # Strong signals: explicitly involve text recognition
        strong_categories = {
            'text_query', 'read_text', 'reading', 'text_location', 'written_query',
            'sign_text', 'recognize', 'ocr_keyword', 'text_recognition',
            'license_plate', 'number_plate', 'plate_number', 'registration',
            'number_query', 'number_read', 'number_location',
            'letter_query', 'letter_read',
            'transcribe', 'copy_text', 'extract',
            'screen_display', 'document_read', 'menu_text',
        }
        return bool(matched_categories & strong_categories)
    
    elif min_confidence == 'medium':
        # Medium signals: exclude categories that may produce false positives
        weak_categories = {'signage', 'words'}  # These may produce false positives
        strong_matches = matched_categories - weak_categories
        return bool(strong_matches)
    
    else:  # low
        return True


def get_ocr_categories(messages: List[dict]) -> Set[str]:
    """Return the OCR categories matched by a sample.

    Returns:
        Set[str]: Set of matched categories
    """
    all_text = ' '.join([msg.get('content', '').lower() for msg in messages])
    
    matched_categories = set()
    for pattern, category in COMPILED_OCR_PATTERNS:
        if pattern.search(all_text):
            matched_categories.add(category)
    
    return matched_categories


# ===== Usage example =====
if __name__ == '__main__':
    test_messages = [
        {"role": "user", "content": "<image>\nWhat does the text on the sign say?"},
        {"role": "assistant", "content": "The sign says 'STOP'."},
    ]
    
    print(f"is_ocr_sample: {is_ocr_sample(test_messages)}")
    print(f"categories: {get_ocr_categories(test_messages)}")
