# Check out the old version without pydantic-ai at
# https://github.com/phi1010/zammad-openai-tool-agent/blob/v1.0/zammad_openai_agent.py
#
# This code currently does not work
# See the bottom of the marimo notebook for an example

import zammad
from zammad import OpenAiTools, BaseModel, Field, ZammadAPI
from config import settings
from time import sleep

tools = OpenAiTools()


class Location(BaseModel):
    """Location"""
    city: str = Field(description="The city name in english")
    country: str = Field(description="The country name in english")


@tools
def get_weather(location: Location):
    print(location)
    return dict(temperature="20°C", humidity="10%")


def run():
    # Note the Host URL should be in this format: 'https://zammad.example.org/api/v1/'
    zclient = ZammadAPI(url='http://localhost:8080/api/v1/', username=settings.zammad_user,
                        password=settings.zammad_pass)
    while True:
        system_prompt = "\n".join([
            "Keep the answers short, only reply to what you were asked.",
            "Only provide information queried from tool functions, don't give other information.",
            "Ask for information missing in tool function parameters.",
            #"Ask for confirmation of the parameters before running a tool function.",
        ])
        zammad.iterate_zammad_openai(zclient, tools, system_prompt)
        try:
            sleep(5)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    run()
