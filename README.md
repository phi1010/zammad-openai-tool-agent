Look at and run zammad_openai_agent.py to configure your customized agent

This imports a marimo notebook zammad.py containing the implementation of the agent and some additional API calls

TODO: The requirements.txt currently contains a lot more requirements than needed. Try to run the code without installing packages first, and install what is missing.

Example on how tools can be defined:

```py
tools = OpenAiTools()

class Location(BaseModel):
    """Location"""
    city: str = Field(description="The city name in english")
    country: str = Field(description="The country name in english")

@tools
def get_weather(location: Location):
    print(location)
    return dict(temperature="20°C", humidity="10%")
```

Similar projects:

Instead of using https://github.com/phi1010/python-openai-tools-functions-dictionary, https://ai.pydantic.dev/ could be used to declare tools and to communicate with OpenAI.
