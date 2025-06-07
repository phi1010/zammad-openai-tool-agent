import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    from zammad_py import ZammadAPI
    import zammad_py
    import marimo as mo
    from dataclasses import dataclass, field
    from openai import OpenAI
    from openai.types.responses.parsed_response import ParsedResponseOutputMessage, ParsedResponseFunctionToolCall
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
    log = logging.getLogger(__name__)
    memory = Memory("cachedir")
    client = OpenAI()


@app.cell
def _():
    logging.basicConfig(level=logging.INFO)
    log.debug("Configured Logging")
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
    zclient = ZammadAPI(url='http://localhost:8080/api/v1/', username=settings.zammad_user, password=settings.zammad_pass)
    zclient
    return (zclient,)


@app.function
def depaginate(page:zammad_py.api.Pagination):
    while page:
        for item in page:
            yield item
        page = page.next_page()


@app.cell
def _():
    refresh_button = mo.ui.refresh(
        options=["10s", "1m", "5m", "10m"],
        default_interval=None,
    )
    refresh_button
    return (refresh_button,)


@app.cell
def _(refresh_button):
    refresh_button.value
    return


@app.cell
def _(refresh_button, zclient):
    refresh_button
    all_tickets=list(depaginate(zclient.ticket.all()))
    all_tickets
    return (all_tickets,)


@app.cell
def _(refresh_button, zclient):
    refresh_button
    fresh_tickets=list(depaginate(zclient.ticket.search("updated_at:>=now-5m")))
    fresh_tickets
    return (fresh_tickets,)


@app.cell
def _(fresh_tickets, zclient):
    {
        ticket["id"]: zclient.ticket.articles(ticket["id"])
        for ticket in fresh_tickets
    }
    return


@app.function
def ticket_to_conversation(zclient, ticket):
    articles = zclient.ticket.articles(ticket["id"])
    conversation = []
    title=ticket["title"]
    match ticket["create_article_sender"]:
        case "Customer":
            conversation.append(
                {
                    "role": "user",
                    "content": title,
                }
            )
        case "Agent":
            conversation.append(
                {
                    "role": "assistant",
                    "content": title,
                }
            )
    for article in articles:
        body = article["body"]
        match article["sender"]:
            case "Customer":
                conversation.append(
                    {
                        "role": "user",
                        "content": body,
                    }
                )
            case "Agent":
                try:
                    data = body
                    if article["content_type"]=="text/html":
                        data = html.unescape(re.sub("<.*?>", " ", body))
                    data = json.loads(data)
                    conversation.append(data)
                except json.JSONDecodeError:
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": body,
                        }
                    )
    return conversation


@app.function
def query_chatgpt_tools(input, tools=[], model="gpt-4.1-mini", seed=1):
    response = client.responses.parse(
        model=model,
        input=input,
        tools=tools,
    )
    return response


@app.cell
def _():


    return


@app.cell
def _(all_tickets, zclient):
    conversations = [
        dict(
            conversation=(conversation := ticket_to_conversation(zclient, ticket)),
            ticket=ticket,
            todo=(
                conversation[-1].get("role", "") == "user"
                or conversation[-1].get("type", "") == "function_call_output"
            ),
        )
        for ticket in all_tickets
    ]
    conversations
    return


@app.cell
def _():


    return


@app.function
def handle_ticket(zclient, ticket, tools, system_prompt):
    conversation = ticket_to_conversation(zclient, ticket)
    todo = (
        conversation[-1].get("role", "") == "user"
        or conversation[-1].get("type", "") == "function_call_output"
    )
    if todo:
        log.info(f"==== Processing Ticket {ticket["id"]} ({ticket["title"]!r}) ====")
        log.info(f"Action required on ticket id {ticket['id']}. Querying OpenAI.")
        log.debug(f"Conversation: {json.dumps(conversation, indent=4)}")
        return query_chatgpt_tools(
            [
                {
                    "role": "developer",
                    "content": system_prompt,
                }
            ]
            + conversation,
            tools=tools.tools,
        )
    else:
        return None


@app.function
def iterate_zammad_openai(zclient, tools, system_prompt):
    fresh_tickets = list(
        depaginate(zclient.ticket.search("updated_at:>=now-5m"))
    )
    for ticket in fresh_tickets:
        response = handle_ticket(zclient, ticket, tools, system_prompt)
        if response:
            for output in response.output:
                handle_response_output(zclient, ticket, output, tools)


@app.cell
def _():
    Port.model_validate({'proto': 'tcp', 'port': 'https', 'target': 'altair.example.com'})
    return


@app.cell
def _():
    system_prompt = (
        """
        You are a support service desk.
        Please only provide the tool calls.
        Please keep the answers short.
        """
        # Ask for confirmation of the parameters before calling the tools.
        """Only call tools if the user instructs you to do so."""
    )
    return (system_prompt,)


@app.cell
def _():
    tools=get_tools()
    return (tools,)


@app.cell
def _(refresh_button, system_prompt, tools, zclient):
    refresh_button
    iterate_zammad_openai(zclient, tools=tools, system_prompt=system_prompt)
    return


@app.function
def handle_response_output(zclient, ticket, output, tools):
    match output:
        case ParsedResponseOutputMessage():
            for content in output.content:
                log.info(f"Received text message from OpenAI: {content.text}")
                zclient.ticket_article.create(
                    params=dict(
                        ticket_id=ticket["id"],
                        # subject="Subject",
                        body=content.text,
                        internal=False,
                        sender="Agent",
                        type="web",
                        content_type="text/plain",
                    )
                )
        case ParsedResponseFunctionToolCall():
            name = output.name
            arguments = json.loads(output.arguments)
            log.info(f"Received tool call from OpenAI: {name}({arguments!r})")
            result = tools.call_function(name, arguments)
            log.info(f"Tool returned result {result!r}")

            call = output.to_dict()
            call.pop("parsed_arguments")
            log.debug("Creating Zammad tool call article")
            zclient.ticket_article.create(
                params=dict(
                    ticket_id=ticket["id"],
                    # subject="Subject",
                    body=json.dumps(call),
                    internal=True,
                    sender="Agent",
                    type="note",
                    content_type="text/plain",
                )
            )
            log.debug("Creating Zammad tool response article")
            zclient.ticket_article.create(
                params=dict(
                    ticket_id=ticket["id"],
                    # subject="Subject",
                    body=json.dumps(
                        {
                            "type": "function_call_output",
                            "call_id": output.call_id,
                            "output": json.dumps(result),
                        },
                    ),
                    internal=True,
                    sender="Agent",
                    type="note",
                    content_type="text/plain",
                )
            )
            log.debug("Articles created.")


@app.cell
def _():


    return


@app.cell
def _():


    return


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


@app.class_definition
class CreateResources(BaseModel):
    """Resources to be created"""

    vms: list[VM] = Field(description="List of Virtual Machines to be created")
    ports: list[Port] = Field(description="List of Ports to be opened")


@app.function
def get_tools():
    tools = OpenAiTools()

    #@tools
    def create_resources(resources: CreateResources):
        for vm in resources.vms:
            print(vm)
        for port in resources.ports:
            print(port)

    @tools
    def open_port(port: Port):
        """Opens a network port"""
        print(port)
        return dict(status="done")

    @tools
    def create_vm(vm: VM):
        """Creates a virtual machine"""
        print(vm)
        return dict(status="running")

    return tools


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
