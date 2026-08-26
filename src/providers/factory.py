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

# Maximum tokens any provider may emit in a single response.
#
# WHY THIS EXISTS. Without a cap, a model that falls into a token-repetition
# loop generates until it exhausts its context window. In the Week 4 run set
# this happened on 9 local-model runs, one of which emitted 82,043 characters
# (the six-character sequence "–" repeated ~13,600 times) and took 30
# MINUTES. Those runs alone are why the local model's mean latency was 79.5s
# against a 4.3s median. They failed either way; the cap changes only how long
# they take to fail.
#
# 2048 is far above anything a legitimate run needs - the longest well-formed
# agent response in the run set is ~700 characters, roughly 200 tokens - so the
# cap is inert on healthy runs and only truncates runaway ones.
#
# It is applied IDENTICALLY to all four providers, which is what keeps it a
# property of the fixed scaffold rather than a per-provider tweak. Anything
# else would break the premise of the experiment.
DEFAULT_MAX_OUTPUT_TOKENS = 2048


def get_llm(provider: str, model: str, temperature: float = 0.0,
            max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, **kwargs):
    """
    Returns a LangChain chat model instance for the given provider.

    provider: one of "openai", "groq", "ollama", "anthropic"
    model: the provider-specific model name (see config/config.yaml)
    temperature: fixed at 0.0 by default across all providers for the main
                 run, since we want to isolate provider effects from
                 sampling randomness. RQ-specific ablations can override this.
    max_output_tokens: uniform generation cap, see DEFAULT_MAX_OUTPUT_TOKENS
                 above. Each provider spells this differently (num_predict on
                 Ollama, max_tokens elsewhere); translating it here is exactly
                 the kind of provider-specific detail this factory exists to
                 absorb. Pass None to disable.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if max_output_tokens is not None:
            kwargs.setdefault("max_tokens", max_output_tokens)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        if max_output_tokens is not None:
            kwargs.setdefault("max_tokens", max_output_tokens)
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("GROQ_API_KEY"),
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama calls the generation cap num_predict, not max_tokens
        if max_output_tokens is not None:
            kwargs.setdefault("num_predict", max_output_tokens)
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
        if max_output_tokens is not None:
            kwargs.setdefault("max_tokens", max_output_tokens)
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
