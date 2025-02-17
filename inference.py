from peft import AutoPeftModelForCausalLM
from transformers import pipeline, AutoTokenizer
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora-quantization",
    low_cpu_mem_usage=True,
    device_map="auto",
)

merged_model = model.merge_and_unload()

pipe = pipeline(
    task="text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.7,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferWindowMemory(
    k=3,  # top 3 nearest conversation
    memory_key="history",
    return_messages=True
)

template = """<|user|>
Answer the question step by step using the following context:

{history}

Current question: {input}

Please provide:
1. Detailed reasoning process
2. Final answer in format "Final Answer: [answer]"
</s>
<|assistant|>
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    response = conversation.run(user_input)
    print(response)
