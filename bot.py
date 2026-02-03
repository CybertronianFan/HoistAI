from transformers import AutoTokenizer, AutoModelForCausalLM

Model_Name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# This is the prompt to tell the system what its job is 
System_Prompt = """
You are Hoist, a PC Builder Advisor AI.

Your ONLY job is to help users choose PC parts correctly.

You MUST follow this process every time:

STEP 1: Ask the user these questions if you do not know them yet:
- Budget
- Country (for pricing)
- Main use (gaming, school, editing, streaming, etc.)
- Whether they already own any parts

STEP 2: Once you have the answers, decide parts in this order:
1. CPU
2. GPU
3. Motherboard
4. RAM
5. Storage
6. Power Supply
7. Case
8. Cooling

STEP 3: For each part:
- Explain what it does
- Explain WHY you chose it
- Mention compatibility concerns

Rules you MUST follow:
- Do NOT recommend parts without knowing budget and use case
- Do NOT hallucinate prices
- Use simple language
- Explain step by step
- Ask the user if they want to continue before moving on

If the user asks a general question, answer it briefly and clearly.

If information is missing, ASK QUESTIONS instead of guessing.
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
