import os
import openai
import string
import numpy as np
import pandas as pd
from gpt_index import (
    GPTKeywordTableIndex,
    GPTSimpleKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    GPTSimpleVectorIndex,
    GPTListIndex,
    readers,
    KeywordExtractPrompt,
    QuestionAnswerPrompt,
    Document
)
from langchain import OpenAI, PromptTemplate

openai.api_key = 'Your_API_KEY'
os.environ['OPENAI_API_KEY'] = 'Your_API_KEY'
path2file = r'path of drk data'


def retrieve_docs(gpt_index, num_documents=10):
    # the function returns a list
    # each index in the list is type gpt_index.readers.schema.base.Document
    query = input("Welche Art von Dokumente wollen Sie?")

    # Default Prompt Template was used with a specified similarity_top_k filter
    response = gpt_index.query(query, similarity_top_k=num_documents, verbose=True, response_mode='default')
    retrieved_docs = []
    for sn in response.source_nodes:
        retrieved_docs.append(readers.Document(sn.source_text))
    return retrieved_docs


def construct_index(documents, model_name='text-davinci-003', save_file=True, save_file_name=None):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens (default)
    num_outputs = 512
    # maximum chunk overlap
    chunk_overlap = 20

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name=model_name, max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap, chunk_size_limit=512)

    # Using Simple Vector Index with default embeddings
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    if save_file:
        if save_file_name is None:
            index.save_to_disk('index.json')
        else:
            index.save_to_disk(save_file_name)

    return index


def search_for_key_words(df, keywords):
    indices_list = []
    texts = df.text.tolist()
    for idx_, t in enumerate(texts):
        for word in keywords:
            word_ = str.lower(word)
            if word_ in str.lower(t):
                indices_list.append(idx_)
                break
    relevant_df = df.iloc[indices_list]
    return relevant_df

def get_gpt_index_documents(df):
    texts = df.text.tolist()
    doc_chunks = []
    temp_ids = []
    for i, text in enumerate(texts):
        _ = f"doc_{i}"#
        temp_ids.append(_)
        doc = Document(text, doc_id=_)
        doc_chunks.append(doc)
    df['doc_id'] = temp_ids
    index = construct_index(doc_chunks)
    return df, index

def get_results_from_df(df, answers):
    # get doc_ids ### option one doc_ids are the indices, opt 2 the doc_id is doc_id_number self
    ids = [ans.doc_id for ans in answers if ans.doc_id in df.index]
    return df[df['doc_id'].isin(ids)]


def main():
    df = pd.read_csv(path2file)
    # adust that key words and saved in a file
    # add a function that reads the file and loads the key words
    key_words = ["obdachlos", "Wohnungslosenhilfen", "Obdachlosenhilfen", "Kältehilfen", "Kältehilfe", "Wärmehilfe",
                 "Winterhilfe", "Wärmebus", "Kältebus", "Obdachbus", "Obdachlosenbus", "Wärmezelt", "Wärmestube",
                 "Kältestube", "Tafeln", "Tafel", "Essensausgabe", "Lebensmittelausgabe", "Suppenküche",
                 "Wohnungslosenberatung", "allgemeine Hilfs- und Weitervermittlungsangebote für Wohnungslose",
                 "Obdachlosenberatung", "Wohnungslosenhilfe", "Obdachlosenhilfe", "Wohnprojekte/Unterbringung",
                 "Übernachtungsheim", "Übernachtung", "Wohnungslosenunterkunft", "Wohnungslosen-Unterkunft",
                 "Wohnungslosenheim", "Obdachlosenheim", "Obdachlosenunterkunft", "Obdachlosen-Unterkunft",
                 "Tagesstätte", "Wohnprojekt", "Hotel+", "HotelPlus", "Hotel Plus", "Weitere assoziierte Angebote",
                 "Sozialstation", "Sozialberatung", "psychosoziale Beratung", "Kleiderkammern/Kleiderläden",
                 "Kleiderladen", "Kleiderkammer", "Second hand", "Second-hand", "Secondhand", "Kinder-Second-hand",
                 "Kinder-Secondhand", "Kiezladen", "Stadtteilladen", "Stadtteil-Laden", "Rotkreuzladen",
                 "Rotkreuz-Laden", "FAIRKaufhaus", "FAIR-Kaufhaus"]
    relevant_df = search_for_key_words(df, key_words)
    relevant_df, index = get_gpt_index_documents(relevant_df)
    # this is just an example query
    query_answer = retrieve_docs(index, num_documents=10)
    results_found = get_results_from_df(df, query_answer)
    results_found.to_excel("results_exp.xlsx", index=False)
    #  manuel check of accuracy.
