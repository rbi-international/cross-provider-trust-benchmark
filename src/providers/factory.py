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

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Claude Sonnet 5 rejects any non-default temperature/top_p/top_k
        # with a 400 error (Anthropic deprecated sampling params on this
        # model generation in favor of adaptive effort control). So unlike
        # every other provider, we deliberately do NOT pass temperature
        # here. This is a genuine methodological asymmetry worth noting
        # in the paper's Threats to Validity section: we can't hold
        # temperature=0 constant across all four providers, because one
        # of them no longer accepts the parameter at all.
        kwargs.pop("temperature", None)
        return ChatAnthropic(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    # watsonx kept as an optional path, uncomment if credentials come together
    # elif provider == "watsonx":
    #     from langchain_ibm import ChatWatsonx
    #     return ChatWatsonx(
    #         model_id=model,
    #         url=kwargs.pop("url", "https://us-south.ml.cloud.ibm.com"),
    #         project_id=os.environ.get("WATSONX_PROJECT_ID"),
    #         apikey=os.environ.get("WATSONX_API_KEY"),
    #         params={"temperature": temperature},
    #         **kwargs,
    #     )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Expected one of: "
            f"openai, groq, ollama, anthropic"
        )
