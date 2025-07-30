import os
from langchain_core.messages import AIMessage, convert_to_messages

DEBUG_AGENTS = os.getenv("DEBUG_AGENTS", "false").lower() == "true"

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        messages = convert_to_messages(node_update.get("messages", []))
        if not messages:
            continue

        if DEBUG_AGENTS:
            print(update_label + "\n")
            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
            print("\n")
            continue

        # If not in debug mode, show only the last plain message
        final_msg = messages[-1]
        if isinstance(final_msg, AIMessage):
            print("\n🚨 EVANA: " + final_msg.content.strip() + "\n")