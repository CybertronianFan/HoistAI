import json
from transformers import AutoTokenizer, AutoModelForCausalLM

with open ("component_database.json") as f:
    components = json.load(f)

Model_Name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# This is the prompt to tell the system what its job is 
System_Prompt = f"""
You are Hoist, a PC Builder Advisor AI.

Your job is to explain to the user their PC parts in detail.

Here are the chosen parts:
CPU: {chosen_cpu['name']} (£{chosen_cpu['price_gbp']})
GPU: {chosen_gpu['name']} (£{chosen_gpu['price_gbp']})
RAM: {chosen_ram['storage_gb']}GB {chosen_ram.get('platform', '')} (£{chosen_ram['price_gbp']})

You MUST do the following for each component:

1. Explain what the component does in simple terms.
2. Explain WHY this component was chosen for this build.
3. Describe how it is compatible with the other components in the build.
4. Explain any potential compatibility concerns.
5. Mention the performance impact of this component in the context of the build.
6. Give step-by-step explanations and avoid technical jargon if possible.

Rules you MUST follow:
- Do NOT hallucinate prices or specs.
- Ask the user questions if something is unclear.
- Do NOT assume anything that is unclear and ask instead. 
- Keep explanations clear and beginner-friendly.
- Always consider how each component interacts with the others.

If the user asks general questions, answer briefly and clearly.
"""


tokenizer = AutoTokenizer.from_pretrained(Model_Name)

# device_map="auto" automatically uses GPU if available, otherwise CPU
model = AutoModelForCausalLM.from_pretrained(
    Model_Name,
    device_map="auto"
)

print("Hi, my name is Hoist and feel free to ask me some questions. Type 'exit' to quit.\n")

# Chat history list to remember previous messages
chat_history = []

while True:
    # Take user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == "exit":
        break

    # Add user input to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Limit chat history to avoid exceeding token limit
    max_history = 6
    recent_history = chat_history[-max_history:]

    # Combine system prompt + chat history for model input
    messages = [{"role": "system", "content": System_Prompt}] + recent_history

    # apply_chat_template converts messages into tokens the model understands
    # return_tensors="pt" returns PyTorch tensors
    # ["input_ids"] selects the actual tensor from the dictionary
    # .to(model.device) moves it to GPU or CPU
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    )["input_ids"].to(model.device)

    # model.generate predicts the next tokens
    # max_new_tokens limits how long the response can be
    # temperature controls creativity 
    # do_sample=True allows random sampling for varied responses
    output_ids = model.generate(
        input_ids,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    # Convert the tensor back into human-readable text
    # skip_special_tokens=True removes internal tokens like <s> or <pad>
    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    # Add AI response to chat history
    chat_history.append({"role": "assistant", "content": response})

    # Print the AI's response
    print("\nHoist:", response, "\n")
