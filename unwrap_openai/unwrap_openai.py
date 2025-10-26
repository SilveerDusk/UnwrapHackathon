# Thank you for participating in the Unwrap Hackathon!
# Free OpenAI access is granted as part of the event to help you build unlimited by cost.
# With great throughput also comes great responsibility! There is an expectation you will not abuse the free credits, as that will hamper our ability to offer similar perks at future events.
# The api keys here will be revoked at the end of the event.


import asyncio
import os
from typing import List, Dict, Optional, Any
from openai import AsyncAzureOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

endpoint = "https://unwrap-hackathon-oct-20-resource.cognitiveservices.azure.com/"


class GPT5Deployment(str, Enum):
    GPT_5_NANO = "gpt-5-nano"  # very fast and cheap for high volume (>1000) tasks that.
    GPT_5_MINI = "gpt-5-mini"  # for when nano doesn't cut it
    GPT_5 = "gpt-5"  # quite expensive, this should be for small N tasks like final answers/analysis.


class ReasoningEffort(str, Enum):
    MINIMAL = "minimal"  # useful for fast, quick decisions
    LOW = "low"  # the most thinking you'll probably need
    MEDIUM = (
        "medium"  # extended for really hard tasks - can take 10s of seconds to return
    )
    HIGH = "high"  # math olympiad type tasks, takes forever and is expensive and you 99% don't need this


subscription_key = os.getenv("SUBSCRIPTION_KEY")

# Semaphore to limit concurrent OpenAI calls to 20
_openai_semaphore = asyncio.Semaphore(20)


async def create_openai_completion(
    messages: List[Dict[str, str]],
    model: GPT5Deployment = GPT5Deployment.GPT_5_NANO,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MINIMAL,
    max_completion_tokens: int = 16384,
    tools: Optional[List[type[BaseModel]]] = None,
    tool_choice: Optional[str | Dict[str, Any]] = None,
    client: Optional[AsyncAzureOpenAI] = None,
) -> ChatCompletion:
    """
    Primary OpenAI call function that uses a semaphore to limit concurrency.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: GPT model deployment to use
        reasoning_effort: Reasoning effort level
        max_completion_tokens: Maximum tokens in completion
        tools: Optional list of Pydantic BaseModel classes to use as tools
        tool_choice: Optional tool choice control ("auto", "none", "required", or specific tool dict)
        client: Optional pre-configured client, creates new one if None

    Returns:
        ChatCompletion response from OpenAI
    """
    async with _openai_semaphore:
        if client is None:
            client = AsyncAzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )

        openai_tools = None
        if tools:
            openai_tools = [pydantic_function_tool(tool) for tool in tools]

        request_params = {
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "model": model.value,
            "reasoning_effort": reasoning_effort,
        }

        if openai_tools:
            request_params["tools"] = openai_tools

            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice

        response = await client.chat.completions.create(**request_params)

        return response


# Example Pydantic tool model
class GetWeatherTool(BaseModel):
    """Get current weather for a location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(
        default="celsius", description="Temperature unit (celsius or fahrenheit)"
    )

    def execute(self) -> Dict[str, Any]:
        """Execute the tool and return weather data."""
        # This would normally call a weather API
        return {
            "location": self.location,
            "temperature": "22°C" if self.unit == "celsius" else "72°F",
            "condition": "sunny",
            "unit": self.unit,
        }


def execute_tool_call(
    tool_call, available_tools: Dict[str, type[BaseModel]]
) -> Dict[str, Any]:
    """
    Execute a tool call using the appropriate Pydantic model.

    Args:
        tool_call: The tool call from OpenAI response
        available_tools: Dict mapping tool names to Pydantic model classes

    Returns:
        Result of the tool execution
    """
    import json

    tool_name = tool_call.function.name
    if tool_name not in available_tools:
        return {"error": f"Tool {tool_name} not found"}

    try:
        # Parse arguments and create tool instance
        args = json.loads(tool_call.function.arguments)
        tool_instance = available_tools[tool_name](**args)

        # Execute the tool if it has an execute method
        if hasattr(tool_instance, "execute"):
            return tool_instance.execute()
        else:
            return {"error": f"Tool {tool_name} does not have an execute method"}

    except Exception as e:
        return {"error": f"Error executing tool {tool_name}: {str(e)}"}


async def example_basic_chat() -> None:
    """Example of basic chat completion without tools."""
    print("=== Example: Basic Chat ===")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        },
    ]

    response = await create_openai_completion(messages)
    print(response.choices[0].message.content)


async def example_auto_tool_selection() -> None:
    """Example of chat with tools where model decides whether to use them."""
    print("\n=== Example: Auto Tool Selection ===")
    messages_with_tools = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to weather information.",
        },
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?",
        },
    ]

    response_with_tools = await create_openai_completion(
        messages=messages_with_tools,
        tools=[GetWeatherTool],
        tool_choice="auto",  # Let the model decide
    )
    print(f"Response: {response_with_tools.choices[0].message.content}")

    # Check if the model wants to use a tool
    if response_with_tools.choices[0].message.tool_calls:
        print("Model requested tool calls:")
        available_tools = {"GetWeatherTool": GetWeatherTool}

        for tool_call in response_with_tools.choices[0].message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")

            # Execute the tool
            result = execute_tool_call(tool_call, available_tools)
            print(f"Tool result: {result}")


async def example_required_tool_usage() -> None:
    """Example of forcing the model to use tools."""
    print("\n=== Example: Required Tool Usage ===")
    messages_forced_tool = [
        {
            "role": "user",
            "content": "Get me some weather data.",
        },
    ]

    response_forced = await create_openai_completion(
        messages=messages_forced_tool,
        tools=[GetWeatherTool],
        tool_choice="required",  # Force tool usage
    )

    print(f"Forced response: {response_forced.choices[0].message.content}")
    if response_forced.choices[0].message.tool_calls:
        print("Forced tool calls:")
        for tool_call in response_forced.choices[0].message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            result = execute_tool_call(tool_call, {"GetWeatherTool": GetWeatherTool})
            print(f"Result: {result}")


async def example_disabled_tools() -> None:
    """Example of explicitly disabling tool usage even when tools are available."""
    print("\n=== Example: Disabled Tools ===")
    messages_no_tools = [
        {
            "role": "user",
            "content": "Tell me about the weather without using any tools.",
        },
    ]

    response_no_tools = await create_openai_completion(
        messages=messages_no_tools,
        tools=[GetWeatherTool],
        tool_choice="none",  # Explicitly disable tools
    )

    print(f"Response without tools: {response_no_tools.choices[0].message.content}")
    print(
        f"Tool calls made: {len(response_no_tools.choices[0].message.tool_calls or [])}"
    )

async def summarize_post(post_content) -> None:
    """Example of basic chat completion without tools."""

    try:

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant, who objectively summarizes a string of inputted text into an array containing the singular topic or a couple of topics of the text in under 3 words.\n\n Example: Advice for renting a vehicle to drive in the Orlando area\nHello everyone, I’ve been driving for Uber in the Orlando area, mostly (Disney and Universal), for a few months using my personal vehicle. I typically drive in the evenings to avoid dealing with to much traffic. My question is has anyone rented a vehicle through Ubers marketplace and is it worth it? I have been on the fence as I do enjoy driving but hate the wear and tear on my personal vehicle. It does however seem very costly to rent and my concern is I will be driving to only afford to keep the rental each week and not actually make money. I understand driving just in the evenings likely won’t be enough but it seems everytime I have tried to driving during the day I make very little due to traffic.\nOne other question I have is it worth renting an electric vehicle? How many hours can you actually drive on a single charge? It seems like having to find a charging station and wait for it to charge would be extremely annoying, costly and time consuming?\nThanks for the advice. Output: [Renting a vehicle, Driver Question, Orlando Area]",
            },
            {
                "role": "user",
                "content": f"Please summarize the following post content:\n\n{post_content}",
            },
        ]

        response = await create_openai_completion(messages)
        return response.choices[0].message.content
    
    except:
        return "[]"
    

async def summarize_comments(text) -> None:
    """Example of basic chat completion without tools."""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant, who objectively compares a posts text to the array of comments on the post and provides a one word sentiment analysis of whether the comments are typically agree, disagree, or neutral towards the post content.\n\n Example: Post: cancelling rides gets rid of surge now?\nis this market dependent? in north NJ and I just lost my surge by accepting and canceling a ride, never has this been the case before.\n\nComments: [\"I noticed that too. I accepted a ride in a 1.8x surge area and then canceled it, and the surge disappeared for me as well.\", \"Yeah, I think it's a new tactic Uber is using to prevent drivers from gaming the system. Kinda annoying though.\", \"I haven't experienced this yet, but it makes sense. Uber wants to ensure that drivers are actually completing rides during surge pricing.\", \"This is frustrating. I rely on surge pricing to make decent money during peak hours, and now it feels like a gamble every time I accept a ride.\", \"I wonder if this is just a temporary glitch or if Uber has officially changed their policy on surge pricing.\", \"I've been driving for a while and haven't seen this happen before. It could be specific to certain areas or times.\", \"It's possible that Uber is trying to discourage drivers from accepting rides just for the surge and then canceling them. Makes sense from their perspective.\", \"I think it's important for drivers to be aware of this change so they can adjust their strategies accordingly.\", \"Has anyone else experienced this in different cities or is it just happening in NJ?\", \"Overall, it seems like Uber is tightening their rules around surge pricing to ensure fairness for both drivers and riders.\"]\n\nOutput: Agree",
            },
            {
                "role": "user",
                "content": f"Please analyze the following post and comments:\n\n{text}",
            },
        ]

        response = await create_openai_completion(messages)
        return response.choices[0].message.content
    except:
        return "N/A"

async def generalize_insights(insight_data):
    """Generalize multiple similar insights into one broader insight"""
    messages = [
        {
            "role": "system",
            "content": """You are an expert at analyzing Reddit discussion patterns and creating broader insights.
            
            Given a list of similar insights with their mention counts, create ONE generalized insight that:
            1. Captures the common theme across all insights
            2. Is broader and more general than the individual insights
            3. Is 3-8 words maximum
            4. Uses general terms that would apply to similar discussions
            5. Focuses on the main topic/theme, not specific details
            
            Examples of good generalized insights:
            - "Driver earnings and pay rates"
            - "Vehicle maintenance and repairs"
            - "Passenger behavior complaints"
            - "App technical issues"
            - "Work schedule and hours"
            - "Safety concerns and incidents"
            - "Ride cancellation problems"
            - "Market saturation concerns"
            
            Return only the generalized insight text, nothing else."""
        },
        {
            "role": "user",
            "content": f"Generalize these similar insights into one broader insight:\n\n{insight_data}"
        },
    ]

    response = await create_openai_completion(messages)
    return response.choices[0].message.content.strip()


# async def main() -> None:
#     """Run all example functions to demonstrate different OpenAI usage patterns."""
#     await example_basic_chat()
#     await example_auto_tool_selection()
#     await example_required_tool_usage()
#     await example_disabled_tools()


# if __name__ == "__main__":
#     asyncio.run(main())
