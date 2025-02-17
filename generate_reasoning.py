from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from datasets import load_dataset

llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-fp16.gguf", # Make sure your model_path is correct
    n_gpu_layers=-1,
    max_tokens=2048,
    n_ctx=2048,
    seed=42,
    verbose=False,
    stop=["</answer>"]
)

template = """
<|user|>
You are a math expert. Please answer the following question by providing a detailed, step-by-step explanation, as if you were explaining to a 6-year-old. All answer must be in Vietnamese. Use clear, simple Vietnamese and appropriate mathematical notation.

A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {input}.
<|assistant|>
"""

prompt_template = PromptTemplate(input_variables=["input"], template=template)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

data = load_dataset("5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated")
data = data["train"].shuffle(seed=42).select(range(1))

def reasoning(example):
    reasoning = llm_chain.invoke({
        "input": example["query_vi"]
    })
    example["reasoning"]=reasoning
    return example
data=data.map(reasoning)
eval_template = """
<|user|>
Là một giáo viên toán, hãy kiểm tra xem *phần lập luận* và *câu trả lời* của học sinh (trong thẻ <answer>) có khớp với *đáp án chuẩn* không. Chỉ trả lời Có hoặc Không.

*Câu hỏi:* {question}

*Lập luận và đáp án của học sinh:*
{student_reasoning}

*Đáp án chuẩn:* {correct_answer}
<|assistant|>
"""

eval_prompt = PromptTemplate(
    input_variables=["question", "student_reasoning", "correct_answer"],
    template=eval_template
)

eval_chain = LLMChain(llm=llm, prompt=eval_prompt)
def evaluate_reasoning(example):
    # Gọi LLM để đánh giá
    eval_response = eval_chain.invoke({
        "question": example["query_vi"],
        "student_reasoning": example["reasoning"],
        "correct_answer": example["response_vi"]
    })

    # Xử lý kết quả
    judgment = eval_response["text"].strip().lower()
    example["is_correct"] = "có" in judgment  # True nếu đúng, False nếu sai
    return example

data = data.map(evaluate_reasoning)
data.to_json("reasoning_data.json")
