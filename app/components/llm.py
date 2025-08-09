from langchain_openai import ChatOpenAI
from app.config.config import OPENAI_API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(model_name: str = "gpt-4o-mini", openai_api_key: str = OPENAI_API_KEY):
    try:
        logger.info(f"Loading LLM from OpenAI using model: {model_name}...")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=256,
        )

        logger.info("LLM loaded successfully from OpenAI.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load an LLM from OpenAI", e)
        logger.error(str(error_message))
        return None