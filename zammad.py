import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    from zammad_py import ZammadAPI
    import zammad_py
    import marimo as mo
    from dataclasses import dataclass, field
    from openai import OpenAI
    from openai.types.responses.parsed_response import (
        ParsedResponseOutputMessage,
        ParsedResponseFunctionToolCall,
    )
    import json
    from enum import Enum
    from pydantic import BaseModel, Field
    from pathlib import Path
    import random
    import requests
    from icecream import ic
    from joblib import Memory
    from openai import OpenAI
    import json
    import polars as pl
    import functools
    import html
    import re
    from xml.etree import ElementTree as ET
    import markdown2
    from openaitools import OpenAiTools
    import os
    import logging
    import sys
    import base64
    import gzip
    import typing
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelMessage
    from pydantic_core import to_jsonable_python

    log = logging.getLogger(__name__)


@app.cell
def _():
    logging.basicConfig(level=logging.INFO)
    log.debug("Configured Logging")
    return


@app.cell
def _():

    agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')


    @agent.tool_plain  
    def roll_die() -> str:
        """Roll a six-sided die and return the result."""
        num=random.randint(1, 6)
        print("Number: "+str(num))
        return str(num)


    @agent.tool  
    def get_player_name(ctx: RunContext[str]) -> str:
        """Get the user's name."""
        return ctx.deps

    return (agent,)


@app.cell
async def _(agent):


    result1 = await agent.run('Tell me a random number.', deps="Phillip")

    return (result1,)


@app.cell
def _():
    return


@app.cell
def _(result1):
    result1, type(result1)
    return


@app.cell
def _(result1):
    result1.all_messages()
    return


@app.cell
def _(result1):
    result1.new_messages()
    return


@app.cell
def _(result1):
    result1.output
    return


@app.cell
async def _(agent, result1):
    result2 = await agent.run('Explain?', message_history=result1.new_messages())
    print(result2)
    print(result2.output)
    return (result2,)


@app.cell
def _(result2):
    result2.new_messages()
    return


@app.cell
def _(result2):
    result2.all_messages()
    return


@app.cell
def _(result2):
    history_step_2 = result2.all_messages()
    as_python_objects2 = to_jsonable_python(history_step_2) 
    as_python_objects2
    return (as_python_objects2,)


@app.cell
def _(as_python_objects2):
    (json_history_2:=json.dumps(as_python_objects2)), len(json_history_2)
    return


@app.cell
def _(as_python_objects2):
    (zipped_json_history_2 := base64.b64encode(gzip.compress(json.dumps(as_python_objects2).encode("utf8")))), len(zipped_json_history_2)
    return


@app.cell
def _(as_python_objects2):
    same_history_as_step_2 = ModelMessagesTypeAdapter.validate_python(
        as_python_objects2
    )
    same_history_as_step_2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Zammad""")
    return


@app.cell
def _():
    os.chdir(Path(__file__).parent)
    from config import settings

    settings
    return (settings,)


@app.cell
def _(settings):
    # Note the Host URL should be in this format: 'https://zammad.example.org/api/v1/'
    zclient = ZammadAPI(
        url="http://localhost:8080/api/v1/",
        username=settings.zammad_user,
        password=settings.zammad_pass,
    )
    zclient
    return (zclient,)


@app.function
def depaginate(page: zammad_py.api.Pagination):
    while page:
        for item in page:
            yield item
        page = page.next_page()


@app.cell
def _(refresh_button):
    refresh_button.value
    return


@app.cell
def _(refresh_button, zclient):
    refresh_button
    all_tickets = list(depaginate(zclient.ticket.all()))
    all_tickets
    return


@app.cell
def _(refresh_button, zclient):
    refresh_button
    fresh_tickets = list(depaginate(zclient.ticket.search("updated_at:>=now-5m")))
    fresh_tickets
    return (fresh_tickets,)


@app.cell
def _(fresh_tickets, zclient):
    {
        ticket["id"]: zclient.ticket.articles(ticket["id"])
        for ticket in fresh_tickets
    }
    return


@app.class_definition
class ParsedTicket(BaseModel):
    conversation : list[ModelMessage]
    ctxdeps : typing.Any 
    query : str | None


@app.function
def parse_ticket(zclient, ticket, ctxdeps_type) -> ParsedTicket:
    articles = zclient.ticket.articles(ticket["id"])
    conversation = []
    title = ticket["title"]
    match ticket["create_article_sender"]:
        case "Customer":
            pass
        case "Agent":
            pass
    ctxdeps = None
    last_query = ""
    for article in articles:
        body = article["body"]
        content_type = article["content_type"]

        if content_type == "text/html":
            data = html.unescape(re.sub("<.*?>", " ", body))
        elif content_type == "text/plain":
            data = body
        else:
            log.error(f"Message with unknown content type {content_type!r}")
            continue

        match article["sender"]:
            case "Customer":
                last_query += f"""
                    <message>
                    {body}
                    </message>
                """
            case "Agent":
                last_query = ""
                if article["internal"] == True:
                    try:
                        data = json.loads(
                            gzip.decompress(base64.b64decode(data)).decode(
                                "utf8"
                            )
                        )
                        # print(data)
                        conversation += data["conversation"]
                        # print(conversation)
                        try:
                            ctxdeps_data = data["ctx"]
                            ctxdeps = (
                                ctxdeps_type.model_validate(ctxdeps_data)
                                if ctxdeps_data is not None
                                else None
                            )
                        except:
                            log.error(
                                f"Failed to parse ctxdeps from value {data['ctx']}"
                            )
                    except:
                        # These are human-written notes
                        log.warn(f"Failed to parse (probably human) internal response: {data}")
                else:
                    # These are responses from the bot or from a human
                    pass
    if not conversation:
        if last_query:
            last_query = (
                f"""
            <title>
            {title}
            </title>
            """
                + last_query
            )
    if last_query:
        log.info(f"Got query:\n{last_query}")
    return ParsedTicket(
        conversation=conversation, ctxdeps=ctxdeps, query=last_query or None
    )


@app.cell(hide_code=True)
def _(fresh_tickets, zclient):
    conversations = [
        dict(
            **(conversation := parse_ticket(zclient, ticket, ctxdeps_type=ContextDeps)).dict(),
            ticket=str(ticket),
        )
        for ticket in fresh_tickets
    ]
    conversations
    return


@app.class_definition
class HandledTicket(BaseModel):
    response : AgentRunResult
    ctxdeps : typing.Any


@app.function
async def handle_ticket(zclient, get_agent, ticket, get_initial_context_deps, ctxdeps_type):
    parsed_ticket : ParsedTicket = parse_ticket(zclient, ticket, ctxdeps_type)
    ctxdeps = parsed_ticket.ctxdeps
    if ctxdeps is None:
        ctxdeps = get_initial_context_deps(zclient, ticket)
    agent = get_agent(ctxdeps, ticket, zclient)
    todo = bool(parsed_ticket.query)
    if todo:
        log.info(
            f"==== Processing Ticket {ticket['id']} ({ticket['title']!r}) ===="
        )
        log.info(
            f"Action required on ticket id {ticket['id']}. Querying OpenAI."
        )
        #log.debug(f"Conversation: {json.dumps([x.dict() for x in parsed_ticket.conversation], indent=4)}")
        message_history = parsed_ticket.conversation
        query = parsed_ticket.query
        response = await agent.run(query, message_history=message_history, deps=ctxdeps)
        return HandledTicket(response=response, ctxdeps=ctxdeps)
    else:
        return None


@app.function(hide_code=True)
async def iterate_zammad_openai(zclient, get_agent, get_initial_context_deps, ctxdeps_type):
    fresh_tickets = list(
        depaginate(zclient.ticket.search("updated_at:>=now-5m"))
    )
    for ticket in fresh_tickets:
        handled_ticket = await handle_ticket(
            zclient,
            get_agent=get_agent,
            ticket=ticket,
            get_initial_context_deps=get_initial_context_deps,
            ctxdeps_type=ctxdeps_type,
        )
        if handled_ticket:
            handle_response_output(
                zclient,
                ticket,
                handled_ticket.response,
                handled_ticket.ctxdeps,
            )


@app.cell(hide_code=True)
def _():
    Port.model_validate(
        {"proto": "tcp", "port": "https", "target": "altair.example.com"}
    )
    return


@app.function(hide_code=True)
def handle_response_output(zclient, ticket, response, ctx):
    if output := response.output:
        log.info(f"Received text message from OpenAI: {output}")
        zclient.ticket_article.create(
            params=dict(
                ticket_id=ticket["id"],
                # subject="Subject",
                body=output,
                internal=False,
                sender="Agent",
                type="web",
                content_type="text/plain",
            )
        )

    log.debug("Creating Zammad response history article")
    conversation_json = json.dumps(
        dict(ctx=ctx.dict(), conversation=to_jsonable_python(response.new_messages()))
    )
    compressed_json = base64.b64encode(
        gzip.compress(conversation_json.encode("utf8"))
    )
    zclient.ticket_article.create(
        params=dict(
            ticket_id=ticket["id"],
            # subject="Subject",
            body=compressed_json,
            internal=True,
            sender="Agent",
            type="note",
            content_type="text/plain",
        )
    )
    log.debug("Articles created.")


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Run""")
    return


@app.cell
def _():
    refresh_button = mo.ui.refresh(
        options=["10s", "1m", "5m", "10m"],
        default_interval=None,
    )
    refresh_button
    return (refresh_button,)


@app.class_definition
class ContextDeps(BaseModel):
    user_name : str


@app.class_definition
class VM(BaseModel):
    hostname: str = Field(description="The machines hostname, in short form")
    memory: int = Field(description="The RAM memory in GB")
    cpus: int = Field(description="The number of cpus")


@app.class_definition
class Port(BaseModel):
    proto: str = Field(description="The protocol, e.g. tcp/udp")
    port: str = Field(description="The port number, e.g. 443")
    target: str = Field(description="The taget hostname as FQDN")


@app.function
def get_tools(agent):
    @agent.tool
    def open_port(ctx: RunContext[ContextDeps], port: Port):
        """Opens a network port"""
        print(port)
        return dict(status="Port was opened.")

    @agent.tool
    def create_vm(ctx: RunContext[ContextDeps], vm: VM):
        """Creates a virtual machine"""
        print(vm)
        return dict(
            status="Request is being processed.",
            next_action="Ask the user if they want to open a port.",
        )


@app.cell
def _():
    system_prompt = (
        """You are a support service desk."""
        """Please only provide the tool calls."""
        """Please keep the answers short."""
        # """Ask for confirmation of the parameters before calling the tools."""
        """Only call tools if the user instructs you to do so."""
    )
    return (system_prompt,)


@app.cell
def _(system_prompt):
    def get_agent(ctx: ContextDeps, ticket: dict, zclient: ZammadAPI):
        agent = Agent(
            "openai:gpt-4.1-mini",
            system_prompt=system_prompt,
            deps=ctx or ContextDeps(user_name="Phillip"),
        )
        get_tools(agent)
        return agent
    return (get_agent,)


@app.function
def get_initial_context_deps(zclient, ticket):
    user = zclient.user.find(ticket["customer_id"])
    user_name = user["firstname"]
    ctxdeps = ContextDeps(user_name=user_name)
    return ctxdeps


@app.cell
async def _(get_agent, refresh_button, zclient):
    refresh_button
    await iterate_zammad_openai(zclient, get_agent=get_agent, get_initial_context_deps=get_initial_context_deps, ctxdeps_type=ContextDeps)
    return


if __name__ == "__main__":
    app.run()
