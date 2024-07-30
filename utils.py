from lang_cleaner.eng import english_cleaners, get_english_symbols

def cleaned_text_to_sequence(cleaned_text: str, lang: str):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if lang == 'en':
    _, _symbol_to_id, _ = get_english_symbols()
  else:
    raise ValueError(f'Language {lang} is not supported.')

  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def text_to_sequence(text: str, lang: str):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if lang == "en":
    cleaner_fn = english_cleaners
    _, _symbol_to_id, _ = get_english_symbols()

  else:
    raise ValueError(f"Language {lang} not supported")
  sequence = []
  if cleaner_fn is not None:
    text = cleaner_fn(text)
  for symbol in text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence
