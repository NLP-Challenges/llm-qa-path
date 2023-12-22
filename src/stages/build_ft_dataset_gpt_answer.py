
import argparse
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_from_disk
import datasets
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def generate_answer(question, context, extracted_answer):
    template = (
        """
        [INST] Du bist DatasetGPT und deine Aufgabe ist es, auf Basis von gegebener FRAGE, KONTEXT und der extrahierten ANTWORT dazu, eine abstrahierte, ausformulierte Antwort zu schreiben. Schreibe die ANTWORT so, dass die Frage verständlich und vollständig beantwortet werden kann, aber nicht zu lang wird. Am Ende der Antwort deklarierst du die QUELLE aus dem verwendeten Kontext im Format (Quelle: VALUE). Benutze nur den Kontext für die Beantwortung, und kein Wissen dass du aus deinen Trainingsdaten hast. [/INST]
        """
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = (
        """
        [FRAGE] {question} [/FRAGE]

        [KONTEXT] 
        {context}
        [/KONTEXT]

        [EXTRAHIERTE ANTWORT] {extracted_answer} [EXTRAHIERTE ANTWORT]

        ABSTRAHIERTE ANTWORT:
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(
        question=question, context=context, extracted_answer=extracted_answer
    ).to_messages()
    response = chat(messages)
    return response.content

def generate_declined_answer(question, context, extracted_answer):
    template = (
        """
        [INST] Du bist DatasetGPT. Dir wurde eine FRAGE gestellt, die Aufgrund der dir verfügbaren Informationen (KONTEXT) nicht beantwortet werden kann. Schreibe eine kurze Antwort. [/INST]
        """
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = (
        """
        [FRAGE] {question} [/FRAGE]

        [KONTEXT] 
        {context}
        [/KONTEXT]

        [EXTRAHIERTE ANTWORT] {extracted_answer} [EXTRAHIERTE ANTWORT]

        ABGELEHNTE ANTWORT:
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(
        question=question, context=context, extracted_answer=extracted_answer
    ).to_messages()
    response = chat(messages)
    return response.content

load_dotenv()

parser = argparse.ArgumentParser()

# Add positional arguments
parser.add_argument('dataset_filename_input')
parser.add_argument('dataset_filename_output')

args = parser.parse_args()

# Access the arguments
dataset_filename_input = args.dataset_filename_input
dataset_filename_output = args.dataset_filename_output

# load germansquad dataset (train split) and convert to pandas dataframe
df = pd.DataFrame(load_from_disk(dataset_filename_input, keep_in_memory=True))

# Setup OpenAI API
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", request_timeout=60)

#rename answers column since this contains the extractive answer
df = df.rename(columns={"answers":"extractive_answer"}, inplace=False)

## Generate abstractive answers
# iterate over rows with iterrows()
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if row["can_be_answered"]:
        df.at[index, "abstractive_answer"] = generate_answer(row["question"], row["sourced_context"], row["extractive_answer"])
    else:
        df.at[index, "abstractive_answer"] = generate_declined_answer(row["question"], row["sourced_context"], row["extractive_answer"])

# save updated dataset
datasets.Dataset.from_pandas(df).save_to_disk(dataset_filename_output)

#wait a sec to avoid simulateous access to files
time.sleep(1)