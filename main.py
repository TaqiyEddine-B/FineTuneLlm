from chat import Chat
from inference import HuggingfaceInference

# def answer_function(query:str):

#     return query


base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_inference = HuggingfaceInference(model_id=base_model)

Chat(answer_function=hf_inference.generate)
