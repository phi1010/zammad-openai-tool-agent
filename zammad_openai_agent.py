#!/usr/bin/env python3
import asyncio
import logging

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Agent
from zammad_py import ZammadAPI

import zammad
from config import settings
from time import sleep

log = logging.getLogger(__name__)


class Location(BaseModel):
    """Location"""
    city: str = Field(description="The city name in english")
    country: str = Field(description="The country name in english")


class ContextDeps(BaseModel):
    user_name: str
    age: str = Field(default=25)


def get_weather(ctx: RunContext[ContextDeps], location: Location):
    """Return weather from given location"""
    print(location)
    return dict(temperature="20°C", humidity="10%")


def get_user_name(ctx: RunContext[ContextDeps]):
    """Returns the users name"""
    print(ctx.deps.user_name)
    return ctx.deps.user_name


def get_user_age(ctx: RunContext[ContextDeps]):
    """Returns the users age"""
    # TODO does not work yet. System prompt?
    print(ctx.deps.age)
    return ctx.deps.age


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


def get_initial_context_deps(zclient, ticket):
    user = zclient.user.find(ticket["customer_id"])
    user_name = user["firstname"]
    ctxdeps = ContextDeps(user_name=user_name)
    return ctxdeps


async def run():
    logging.basicConfig(level=logging.DEBUG)
    log.debug("Configured Logging")
    # Note the Host URL should be in this format: 'https://zammad.example.org/api/v1/'
    zclient = ZammadAPI(url='http://localhost:8080/api/v1/', username=settings.zammad_user,
                        password=settings.zammad_pass)
    while True:
        await zammad.iterate_zammad_openai(
            zclient=zclient,
            get_agent=get_agent,
            get_initial_context_deps=get_initial_context_deps,
            ctxdeps_type=ContextDeps,
        )
        try:
            sleep(5)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    asyncio.run(run())
