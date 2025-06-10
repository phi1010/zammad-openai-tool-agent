Look at and run zammad_openai_agent.py to configure your customized agent

This imports a marimo notebook zammad.py containing the implementation of the agent and some additional API calls

TODO: The requirements.txt currently contains a lot more requirements than needed. Try to run the code without installing packages first, and install what is missing.

Example on how tools can be defined:

```py

class Location(BaseModel):
    """Location"""
    city: str = Field(description="The city name in english")
    country: str = Field(description="The country name in english")


def get_weather(ctx: RunContext[ContextDeps], location: Location):
    """Return weather from given location"""
    print(location)
    return dict(temperature="20°C", humidity="10%")


def get_agent(ctxdeps: ContextDeps, ticket: dict, zclient: ZammadAPI):
    system_prompt = "\n".join([
        "Greet the user using their name on the first reply.",
        "Keep the answers short, only reply to what you were asked.",
        "Only provide information queried from tool functions, don't give other information.",
        "Ask for information missing in tool function parameters.",
        f"The users name is {ctxdeps.user_name!r}." if ctxdeps else "",
        # "Ask for confirmation of the parameters before running a tool function.",
    ])
    logging.info(f"Got context deps {ctxdeps!r}, creating new agent...")
    agent = Agent("openai:gpt-4.1-mini", system_prompt=system_prompt)
    agent.tool(get_weather)
    agent.tool(get_user_name)
    agent.tool(get_user_age)
    return agent
```
