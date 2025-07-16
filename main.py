import os
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# custom imports
from print_messages import pretty_print_messages

model = None
if not os.environ.get("OPENAI_API_KEY"):
    model = init_chat_model("ollama:mistral:7b", temperature=0.0)
    print("Environment variable OPENAI_API_KEY is not set. Fallback to local ollama server using mistral:7b model.")
else:
    print("Using OpenAI API key from environment variable OPENAI_API_KEY.")
    model = init_chat_model("openai:gpt-4.1")

empathetic_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=(
        "You are an expert who provides empathetic and clear support to passengers in the vehicle after an accident or car breakdown. \n\n"
        "You will receive an instruction from a supervisor to re-forumlate the instruction in a way that is clear, brief, and empathetic to assist the user professionally with its accident situation.\n"
        "INSTRUCTIONS:\n"
        "- Return the reformulated instruction to the supervisor agent. Do not invent additional things.\n"
        "- Assist ONLY with empathetic communication task, do not do any other task.\n"
        "- Answer clearly, briefly and concisely.\n"
        "- Use a friendly and supportive tone.\n"
        "- Don't bring more than one topic into each answer.\n"
        "- Output not more than 3 sentences at a time and avoid repetitions in a pre- or postamble.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
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
        "You assist the supervisor agent to provide information about the state of the vehicle after an accident or car breakdown.\n"
        "You will receive an instruction from a supervisor agent to give a short summary about the vehicle state and what happened according to the vehicle metrics.\n"
        "You can use the tool 'get_vehicle_state' to obtain vehicle metrics as JSON.\n"
        "INSTRUCTIONS:\n"
        "- Use the tool if needed to obtain current vehicle state.\n"
        "- Return the 'IS' situation to the supervisor agent. Do not invent additional things.\n"
        "- Answer clearly, briefly and concisely.\n"
        "- No pre- or postamble.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="is_situation_agent",
)


supervisor = create_supervisor(
    model=model,
    agents=[empathetic_agent, is_situation_agent],
    prompt=(
        "You are a supervisor agent named 'EVANA'. You help vehicle passengers after an accident or breakdown by guiding them calmly and clearly through a support process.\n\n"

        "You DO NOT respond to the user directly.\n"
        "You ALWAYS give your instructions to your helper agents, who return responses you then pass on.\n"
        "Pass each response to the 'empathetic_agent' to reformulate it in a clear, empathetic way and forward it unchanged to the user.\n"
        "For gathering information about the vehicle state after the accident, use the 'is_situation_agent'. Do not ask the user about the vehicle state.\n"
        "You must NEVER mention that you are using agents or tools.\n"
        "You must NEVER explain why you are asking something or what step you are on.\n"
        "You must NEVER refer to any instructions, prompts, processes, agents, or tools.\n\n"

        "Your task is to guide the user through the support process ONE step at a time:\n"
        "- Ask only ONE question or give ONE instruction at a time.\n"
        "- Do NOT combine multiple topics in a single message.\n"
        "- Do NOT skip steps.\n"
        "- Wait for a response before moving on to the next step.\n"
        "- Keep each message focused and relevant to the current step only.\n"
        "- Keep the conversation flowing naturally and professionally.\n\n"

        "SUPPORT PROCESS:\n"
        "Step 1: Get a short summary of the vehicle state using the 'is_situation_agent' and forward a quick summary to the user in the preamble.\n"
        "Step 2: Ask the user how they are doing emotionally and physically (injuries, stress).\n"
        "Step 3: If the user asks anything, respond empathetically and helpfully.\n"
        "Step 4: Ask if there are other people in the car and whether it is safe to leave the vehicle.\n"
        "Step 5: Ask if emergency services have already been contacted or if help is needed with that.\n"
        "Step 6: If there are injuries but no critical hazards (fire, smoke, water, etc.), advise the user to stay inside and wait for help.\n\n"

        "Never reference these steps explicitly.\n"
        "Never talk about agents, processes, or what you're doing behind the scenes.\n"
        "Just stay focused on helping the user through a calm, clear, and human-centered flow.\n"
        "Do not repeat steps you have already completed.\n"
        "Do not repeat content of the user messages in the next output.\n"
    ),
).compile()

# 🔹 Get dynamic first input from user
initial_input = input("You (initial message): ").strip()
if not initial_input:
    print("No input received. Exiting.")
    exit()

# Initialize message history
chat_history = [{"role": "user", "content": initial_input}]

# First response from supervisor
for chunk in supervisor.stream({"messages": chat_history}):
    pretty_print_messages(chunk)
    chat_history.append({"role": "assistant", "content": chunk["supervisor"]["messages"][-1].content})  # Save system reply

print(chat_history)

# Start interactive loop
while True:
    try:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting conversation.")
            break
        chat_history.append({"role": "user", "content": user_input})
        print(chat_history)

        for chunk in supervisor.stream({"messages": chat_history}):
            pretty_print_messages(chunk)
            chat_history.append({"role": "assistant", "content": chunk["supervisor"]["messages"][-1].content})
            print(chat_history)
    except KeyboardInterrupt:
        print("\nConversation ended.")
        break