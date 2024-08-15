from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from deep_translator import GoogleTranslator

class TranslationInput(BaseModel):
    text: str = Field(description="The text to translate")
    dest_lang: str = Field(description="The target language code (e.g., 'en' for English)")

def translate_text(text: str, dest_lang: str) -> str:
    translator = GoogleTranslator(target=dest_lang)
    translated_text = translator.translate(text)
    return translated_text

translate_text_tool = StructuredTool.from_function(
    func=translate_text,
    name="translate_text_tool",
    description="Translate text to a specified language",
    args_schema=TranslationInput,
    return_direct=True,
)
