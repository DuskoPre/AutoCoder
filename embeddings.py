#  Copyright (c) 19.05.2024 [D. P.] aka duskop; after the call a day after from a Japanese IPO-agency, i'm adding my patreon ID: https://www.patreon.com/florkz_com
#  All rights reserved.

import os
import csv
import shutil
from g4f.client import Client
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
import time

# Heavily derived from OpenAi's cookbook example

load_dotenv()

# the dir is the ./playground directory
REPOSITORY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "playground")

class Embeddings:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

        self.DOC_EMBEDDINGS_MODEL = 'text-ada-001'
        self.QUERY_EMBEDDINGS_MODEL = 'text-ada-001'

        self.SEPARATOR = "\n* "

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.separator_len = len(self.tokenizer.tokenize(self.SEPARATOR))

    def compute_repository_embeddings(self):
        playground_data_path = os.path.join(self.workspace_path, 'playground_data')
        print(f"Playground data path: {playground_data_path}")

        if not any(os.scandir(REPOSITORY_PATH)):
            print(f"No files found in the repository path: {REPOSITORY_PATH}")
            return

        try:
            for filename in os.listdir(playground_data_path):
                file_path = os.path.join(playground_data_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")

        info = self.extract_info(REPOSITORY_PATH)
        self.save_info_to_csv(info)

        csv_path = os.path.join(self.workspace_path, 'playground_data/repository_info.csv')
        print(f"CSV path: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        df = df.set_index(["filePath", "lineCoverage"])
        self.df = df

        context_embeddings = self.compute_doc_embeddings(df)
        embeddings_csv_path = os.path.join(self.workspace_path, 'playground_data/doc_embeddings.csv')
        self.save_doc_embeddings_to_csv(context_embeddings, df, embeddings_csv_path)

        try:
            self.document_embeddings = self.load_embeddings(embeddings_csv_path)
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")

    def extract_info(self, REPOSITORY_PATH):
        info = []
        LINES_PER_CHUNK = 60

        for root, dirs, files in os.walk(REPOSITORY_PATH):
            for file in files:
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        contents = f.read()
                    except:
                        continue
                
                lines = contents.split("\n")
                lines = [line for line in lines if line.strip()]
                chunks = [
                    lines[i:i+LINES_PER_CHUNK]
                    for i in range(0, len(lines), LINES_PER_CHUNK)
                ]
                for i, chunk in enumerate(chunks):
                    chunk = "\n".join(chunk)
                    first_line = i * LINES_PER_CHUNK + 1
                    last_line = first_line + len(chunk.split("\n")) - 1
                    line_coverage = (first_line, last_line)
                    info.append((file_path, line_coverage, chunk))
            
        return info

    def save_info_to_csv(self, info):
        os.makedirs(os.path.join(self.workspace_path, "playground_data"), exist_ok=True)
        csv_filepath = os.path.join(self.workspace_path, 'playground_data/repository_info.csv')
        print(f"Saving info to CSV at: {csv_filepath}")

        with open(csv_filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filePath", "lineCoverage", "content"])
            for file_path, line_coverage, content in info:
                writer.writerow([file_path, line_coverage, content])

    def get_relevant_code_chunks(self, task_description: str, task_context: str):
        query = task_description + "\n" + task_context
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(query, self.document_embeddings)
        selected_chunks = []
        for _, section_index in most_relevant_document_sections:
            try:
                document_section = self.df.loc[section_index]
                selected_chunks.append(self.SEPARATOR + document_section['content'].replace("\n", " "))
                if len(selected_chunks) >= 2:
                    break
            except:
                pass

        return selected_chunks

    def get_embedding(self, text: str, model: str) -> list[float]:
        client = Client()
        result = client.embeddings.create(model=model,
        input=text)
        return result.data[0].embedding

    def get_doc_embedding(self, text: str) -> list[float]:
        return self.get_embedding(text, self.DOC_EMBEDDINGS_MODEL)

    def get_query_embedding(self, text: str) -> list[float]:
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def compute_doc_embeddings(self, df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
        embeddings = {}
        for idx, r in df.iterrows():
            time.sleep(1)
            embeddings[idx] = self.get_doc_embedding(r.content.replace("\n", " "))
        return embeddings

    def save_doc_embeddings_to_csv(self, doc_embeddings: dict, df: pd.DataFrame, csv_filepath: str):
        if len(doc_embeddings) == 0:
            print("No embeddings to save.")
            return

        EMBEDDING_DIM = len(list(doc_embeddings.values())[0])
        print(f"Saving doc embeddings to CSV at: {csv_filepath}")

        embeddings_df = pd.DataFrame(columns=["filePath", "lineCoverage"] + [f"{i}" for i in range(EMBEDDING_DIM)])

        for idx, _ in df.iterrows():
            embedding = doc_embeddings[idx]
            row = [idx[0], idx[1]] + embedding
            embeddings_df.loc[len(embeddings_df)] = row

        embeddings_df.to_csv(csv_filepath, index=False)

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(self, query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        query_embedding = self.get_query_embedding(query)
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        return document_similarities
    
    def load_embeddings(self, fname: str) -> dict[tuple[str, str], list[float]]:
        if not os.path.exists(fname):
            print(f"Embeddings file not found: {fname}")
            return {}

        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "filePath" and c != "lineCoverage"])
        return {
            (r.filePath, r.lineCoverage): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }

if __name__ == "__main__":
    workspace_path = os.path.dirname(os.path.realpath(__file__))
    print(f"Workspace path: {workspace_path}")
    
    embeddings = Embeddings(workspace_path)
    embeddings.compute_repository_embeddings()
