import os
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# custom imports
from print_messages import pretty_print_messages

model = None
if not os.environ.get("OPENAI_API_KEY"):
    model = init_chat_model("ollama:mistral:7b", temperature=0.0, input_token_limit=32768)
    print("Environment variable OPENAI_API_KEY is not set. Fallback to local ollama server using mistral:7b model.")
else:
    print("Using OpenAI API key from environment variable OPENAI_API_KEY.")
    model = init_chat_model("openai:gpt-4.1")

empathetic_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=(
"""You turn supervisor messages into calm, supportive user messages.

RULES:
- Use 2–3 short, clear, empathetic sentences.
- Speak warmly and clearly.
- Keep the meaning and content of the original message, but rephrase it gently.
- Never add additional information or context.
- Never refer to tools, agents, or instructions.
- Focus on one topic per message.
- Return the final message directly to the supervisor.

Your job is to be the human-facing voice of EVANA in stressful moments."""
    ),
    name="empathetic_agent",
)

@tool
def get_vehicle_state() -> dict:
    """Get the current vehicle state as a JSON object."""
    # Return empty JSON for now; simulate real data later
    return {
        "airbags_activated": True,
        "motor_engine_status": "off",
        "door_status": "closed",
        "windows_status": "closed",
        "hazzard_lights_status": "off",
    }

is_situation_agent = create_react_agent(
    model=model,
    tools=[get_vehicle_state],
    prompt=(
"""You receive instructions from the supervisor.

Use the `get_vehicle_state()` tool to get vehicle data.

Output:
- One clear sentence summarizing the situation, e.g.:
“You had an accident. The motor engine is off and airbags are activated.”
- Do NOT add explanations, empathy, greetings, or next steps.
- Return only the final sentence, nothing else.

Your only job is to summarize the vehicle condition in one neutral sentence."""
    ),
    name="is_situation_agent",
)


supervisor = create_supervisor(
    model=model,
    agents=[empathetic_agent, is_situation_agent],
    prompt=(
"""You are EVANA, the coordinator agent.

You NEVER speak to the user. You ONLY:
- Ask `situation_agent` for a 1-sentence vehicle state summary.
- Combine that summary with the first user-facing instruction:
→ Ask if the user is injured or how they are feeling.
- Send that combined message to `empathetic_agent`.
- Output always what `empathetic_agent` returns.

Afterward, proceed step-by-step with the process:
- Ask ONE question or give ONE instruction at a time.
- Each message must go through `empathetic_agent`.
- Never mention tools, agents, or internal steps.
- Wait for the user to respond before continuing.
- Never repeat completed steps.

Trigger phrases like “help”, “emergency”, or “Hey EVANA” begin the process.

Your goal: guide the user through the emergency, calmly and clearly."""
    ),
).compile(checkpointer=checkpointer)

# 🔹 Get dynamic first input from user
initial_input = input("You (initial message): ").strip()
if not initial_input:
    print("No input received. Exiting.")
    exit()

# First response from supervisor
for chunk in supervisor.stream({"messages": [{"role": "user", "content": initial_input}]}, {"configurable": {"thread_id": "1"}},):
    pretty_print_messages(chunk)

# Start interactive loop
while True:
    try:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting conversation.")
            break

        for chunk in supervisor.stream({"messages": [{"role": "user", "content": user_input}]}, {"configurable": {"thread_id": "1"}},):
            pretty_print_messages(chunk)

    except KeyboardInterrupt:
        print("\nConversation ended.")
        break