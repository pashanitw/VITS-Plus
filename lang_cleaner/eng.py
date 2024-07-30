import re
from unidecode import unidecode
from phonemizer.backend import EspeakBackend

def get_english_symbols():

  _pad = '_'
  _punctuation = ';:,.!?¡¿—…"«»“” '
  _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
  _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

  symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

  # Mappings from symbol to numeric ID and vice versa:
  _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  _id_to_symbol = {i: s for i, s in enumerate(symbols)}

  return symbols, _symbol_to_id,  _id_to_symbol

global_phonemizer = EspeakBackend(
    "en-us",
    preserve_punctuation=True,
    with_stress=True
)

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

def convert_to_ascii(text):
  return unidecode(text)

def lowercase(text):
  return text.lower()


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = global_phonemizer.phonemize([text], strip=True)[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes
