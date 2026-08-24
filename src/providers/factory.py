"""
Single factory function that returns a LangChain-compatible chat model for
any of the four providers. This is the ONE place provider-specific code
lives. The agent scaffold (src/agent/scaffold.py) never imports a provider
directly, it only ever calls get_llm(provider_name, model_name) and gets
back an object with the standard LangChain .bind_tools() / .invoke()
interface. That's what makes "same scaffold, different backend" a true
apples-to-apples comparison rather than four different code paths.
"""
import os


def get_llm(provider: str, model: str, temperature: float = 0.0, **kwargs):
    """
    Returns a LangChain chat model instance for the given provider.

    provider: one of "openai", "groq", "ollama", "watsonx"
    model: the provider-specific model name (see config/config.yaml)
    temperature: fixed at 0.0 by default across all providers for the main
                 run, since we want to isolate provider effects from
                 sampling randomness. RQ-specific ablations can override this.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("GROQ_API_KEY"),
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            **kwargs,
        )

    elif provider == "watsonx":
        from langchain_ibm import ChatWatsonx
        return ChatWatsonx(
            model_id=model,
            url=kwargs.pop("url", "https://us-south.ml.cloud.ibm.com"),
            project_id=os.environ.get("WATSONX_PROJECT_ID"),
            apikey=os.environ.get("WATSONX_API_KEY"),
            params={"temperature": temperature},
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Expected one of: "
            f"openai, groq, ollama, watsonx"
        )
