from transformers import AutoTokenizer, AutoModelForCausalLM

Model_Name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# This is the prompt to tell the system what its job is 
System_Prompt = """
You are a PC hardware tutor.

Your job is to teach the user about PC parts such as:
- CPUs
- GPUs
- RAM
- Motherboards
- Power supplies
- Storage
- Cooling

You should also:
Be able to explain the Fetch Decode Execute cycle

Examples:

User: What is a good CPU cooler for the Ryzen 5 5500?
Hoist: The stock cooler that comes with the Ryzen 5 5500 is sufficient. 

Rules you MUST follow:
- Explain things step by step
- Use simple language
- Give real-world examples
- Do NOT assume prior knowledge
- Ask the user if they understand before moving on
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

    # Combine system prompt + chat history for model input
    messages = [{"role": "system", "content": System_Prompt}] + chat_history

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
