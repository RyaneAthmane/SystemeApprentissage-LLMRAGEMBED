# ce code fournit une fonction get_prompt_template qui crée un modèle de prompt en fonction
# du type spécifié et des options de contexte. Le modèle de prompt et la mémoire de conversation
# associée peuvent être utilisés pour générer des prompts cohérents lors de l'interaction
# avec un utilisateur ou un système de dialogue.

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


system_prompt = """  """


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(
                input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(
                input_variables=["context", "question"], template=prompt_template)
    else:
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(
                input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(
                input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(
        input_key="question", memory_key="history")

    return (
        prompt,
        memory,

    )
