# About LLM Tools:
Please see the `llm_agents` + `llm_messages` + `llm_reasoning` + `llm_structured_output` learning_resources directories.

LLM tools are computer functions attached to input_messages sent to and processed by a transformer model, in order to influence the type and nature of the sequence predictions the large language model makes. This is achieved through the design of the design of the LLM's `message' , response`, `input` or `dialogue` API AND typically, the `structured_output` API AND the design of `instructions` inserted into a SYSTEM_PROMPT in the INPUT_MESSAGE.

# Common Marketing Terminology: `Agentic AI`
Using the `dialogue` API + tools (computer functions) + instructions (system prompts) + structured_output APIs with large language models is commonly marketed as `agentic ai`. 

# Don't be confused. Agentic AI is an LLM that has:
* Maybe been trained on a mult-turn and/or multi-turn reasoning dataset and/or a tool-calling dataset.
* An API designed to allow users to insert computer functions into the input_message sent to the LLM.

# What matters / is worth studying with LLM tool-calling?
What is the appropriate/relevant/optimal design for the ---:
* Instructions given to the model via the system_prompt?
* Software workflow that will result in the desired behavior? Or, that will achieved a stated goal of the system designer or organization using the system?
* When do we really need to use an LLM tool? What are the tradeoffs in terms of costs and benefits compared to using static methods?
