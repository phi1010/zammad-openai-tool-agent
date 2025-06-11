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


class ContextDeps(BaseModel):
    user_name: str
    age: int = Field(default=25)


class Location(BaseModel):
    """Location"""
    city: str = Field(description="The city name, only in english language")
    country: str = Field(description="The country name, only in english language")


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


class VM(BaseModel):
    hostname: str = Field(description="The machines hostname, in short form")
    memory: int = Field(description="The RAM memory in GB")
    cpus: int = Field(description="The number of cpus")

class Port(BaseModel):
    proto: str = Field(description="The protocol, e.g. tcp/udp")
    port: str = Field(description="The port number, e.g. 443")
    target: str = Field(description="The taget hostname as FQDN")


def open_port(ctx: RunContext[ContextDeps], port: Port):
    """Opens a network port"""
    print(port)
    return dict(status="Port was opened.")


def create_vm(ctx: RunContext[ContextDeps], vm: VM):
    """Creates a virtual machine"""
    print(vm)
    return dict(
        status="Request is being processed.",
        next_action="Ask the user if they want to open a port.",
    )


def get_agent(ctxdeps: ContextDeps, ticket: dict, zclient: ZammadAPI):
    system_prompt = "\n".join([
        "You are a helpful assistant.",
        "Greet the user using their name on the first reply.",
        "Keep the answers short, only reply to what you were asked.",
        "Only provide information queried from tool functions, don't give other information.",
        "Ask for information missing in tool function parameters.",
        f"The users name is {ctxdeps.user_name!r}." if ctxdeps else "",
        "Your output must strictly be provided in html format. Don't use any buttons or other interactive elements.",
        "Ask for confirmation of the parameters before running a tool function that alters data or executes an action.",
    ])
    logging.info(f"Got context deps {ctxdeps!r}, creating new agent...")
    agent = Agent("openai:gpt-4.1-mini", system_prompt=system_prompt)
    agent.tool(get_weather)
    agent.tool(get_user_name)
    agent.tool(get_user_age)
    agent.tool(open_port)
    agent.tool(create_vm)
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
