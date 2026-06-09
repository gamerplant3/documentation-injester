import os

import cohere
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_cohere_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY environment variable not set. "
                "Add it to .env or export it before running the worker."
            )
        _client = cohere.ClientV2(api_key=api_key)
    return _client
