from typing import cast

import chainlit as cl
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

from inference import BaseInference, FinetunedModelInference


@cl.on_chat_start
async def on_chat_start():
    testing_finetuned_model = True

    base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    experiment_id = "20241106_0551"

    if testing_finetuned_model:
        adapter_path = f"output/Meta-Llama-3.1-8B-Instruct_finetune_{experiment_id}"
        inference = FinetunedModelInference(base_model_name=base_model, adapter_path=adapter_path)
        model = HuggingFacePipeline(pipeline=inference.pipeline)
    else:
        inference = BaseInference(model_id=base_model)
        model = HuggingFacePipeline(pipeline=inference.pipeline)

    template = """<|begin_of_text|><|start_header_id|>
        You are Question/answer assistant.
        Your answer must begin with capital letter and end with full stop.
        <|end_header_id|>{question}<|eot_id|>
        """
    prompt = PromptTemplate.from_template(template)
    runnable = prompt | model
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
