import json
import random
import pandas as pd
from typing import Iterator, List, Tuple, Optional, Union
import gzip
import os


rand_seed = None


def _parse_squad_data(raw):
    dataset = {"ki_text": [], "qas": []}

    for k_id, data in enumerate(raw["data"]):
        article = []
        for p_id, para in enumerate(data["paragraphs"]):
            article.append(para["context"])
            for qa in para["qas"]:
                ques = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                dataset["qas"].append(
                    {
                        "title": data["title"],
                        "paragraph_index": tuple((k_id, p_id)),
                        "question": ques,
                        "answers": answers,
                    }
                )
        dataset["ki_text"].append(
            {"id": k_id, "title": data["title"], "paragraphs": article}
        )

    return dataset


def squad(
    filepath: str,
    max_knowledge: Optional[int] = None,
    max_paragraph: Optional[int] = None,
    max_questions: Optional[int] = None,
) -> Tuple[List[str], Iterator[Tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @param max_knowledge: maximum number of docs in dataset
    @param max_paragraph:
    @param max_questions:
    @return: knowledge list, question & answer pair list
    """
    # Open and read the JSON file
    with open(filepath, "r") as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = _parse_squad_data(data)

    print(
        "max_knowledge",
        max_knowledge,
        "max_paragraph",
        max_paragraph,
        "max_questions",
        max_questions,
    )

    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = (
        max_knowledge
        if max_knowledge is not None and max_knowledge < len(parsed_data["ki_text"])
        else len(parsed_data["ki_text"])
    )
    max_paragraph = max_paragraph if max_knowledge == 1 else None

    # Shuffle the Articles and Questions
    if rand_seed is not None:
        random.seed(rand_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])

    k_ids = [i["id"] for i in parsed_data["ki_text"][:max_knowledge]]

    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data["ki_text"][:max_knowledge]:
        max_para = (
            max_paragraph
            if max_paragraph is not None and max_paragraph < len(article["paragraphs"])
            else len(article["paragraphs"])
        )
        text_list.append(article["title"])
        text_list.append("\n".join(article["paragraphs"][0:max_para]))

    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [
        qa["question"]
        for qa in parsed_data["qas"]
        if qa["paragraph_index"][0] in k_ids
        and (max_paragraph is None or qa["paragraph_index"][1] < max_paragraph)
    ]
    answers = [
        qa["answers"][0]
        for qa in parsed_data["qas"]
        if qa["paragraph_index"][0] in k_ids
        and (max_paragraph is None or qa["paragraph_index"][1] < max_paragraph)
    ]

    dataset = zip(questions, answers)

    return text_list, dataset


def hotpotqa(
    filepath: str, max_knowledge: Optional[int] = None
) -> Tuple[List[str], Iterator[Tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @param max_knowledge:
    @return: knowledge list, question & answer pair list
    """
    # Open and read the JSON
    with open(filepath, "r") as file:
        data = json.load(file)

    if rand_seed is not None:
        random.seed(rand_seed)
        random.shuffle(data)

    questions = [qa["question"] for qa in data]
    answers = [qa["answer"] for qa in data]
    dataset = zip(questions, answers)

    if max_knowledge is None:
        max_knowledge = len(data)
    else:
        max_knowledge = min(max_knowledge, len(data))

    text_list = []
    for _, qa in enumerate(data[:max_knowledge]):
        context = qa["context"]
        context = [c[0] + ": \n" + "".join(c[1]) for c in context]
        article = "\n\n".join(context)

        text_list.append(article)

    return text_list, dataset


def kis(filepath: str) -> Tuple[List[str], Iterator[Tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @return: knowledge list, question & answer pair list
    """
    df = pd.read_csv(filepath)
    dataset = zip(df["sample_question"], df["sample_ground_truth"])
    text_list = df["ki_text"].to_list()

    return text_list, dataset


def natural_questions(filepath, max_knowledge=3, max_question=5, split=None, split_ratio=(0.8, 0.1, 0.1), random_seed=None, chunk_index=0):
    """
    Load Google Natural Questions dataset from gzipped JSONL format.
    
    Args:
        filepath: Path to gzipped JSONL file
        max_knowledge: Maximum number of knowledge items to keep
        max_question: Maximum number of questions to keep per knowledge item
        split: Dataset split to use ('train', 'test', 'eval')
        split_ratio: Ratio for train/test/eval split if split is None
        random_seed: Random seed for shuffling
        chunk_index: Index of the chunk to process (0-based)
        
    Returns:
        List of (text, questions, answers) tuples
    """
    def truncate_text(text, max_tokens=256):
        """Truncate text to max_tokens by splitting on spaces"""
        words = text.split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens])
        return text

    # Initialize counters and lists
    knowledge_count = 0
    question_count = 0
    texts = []
    questions = []
    answers = []
    
    # Calculate chunk boundaries (assuming 100MB chunks)
    chunk_size = 100 * 1024 * 1024  # 100MB in bytes
    start_byte = chunk_index * chunk_size
    end_byte = start_byte + chunk_size
    
    # Process gzipped file
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        # Skip to the start of the chunk
        current_byte = 0
        while current_byte < start_byte:
            line = f.readline()
            if not line:
                break
            current_byte += len(line.encode('utf-8'))
        
        # Process the chunk
        while current_byte < end_byte:
            line = f.readline()
            if not line:
                break
                
            current_byte += len(line.encode('utf-8'))
            
            try:
                data = json.loads(line.strip())
                text = truncate_text(data['document_text'])
                question = data['question_text']
                
                answer = None
                if data['annotations'] and len(data['annotations']) > 0:
                    annotation = data['annotations'][0]
                    if annotation['long_answer']:
                        start = annotation['long_answer']['start_token']
                        end = annotation['long_answer']['end_token']
                        answer = ' '.join(text.split()[start:end])
                    elif annotation['short_answers']:
                        short_answer = annotation['short_answers'][0]
                        start = short_answer['start_token']
                        end = short_answer['end_token']
                        answer = ' '.join(text.split()[start:end])
                
                if answer:
                    if knowledge_count >= max_knowledge:
                        break
                        
                    if question_count >= max_question:
                        question_count = 0
                        knowledge_count += 1
                        if knowledge_count >= max_knowledge:
                            break
                    
                    texts.append(text)
                    questions.append(question)
                    answers.append(answer)
                    question_count += 1
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    print(f"Processed chunk {chunk_index}: {len(texts)} documents with {len(questions)} questions")
    return texts, list(zip(questions, answers)), None


def trivia_qa(filepath, max_knowledge=None, max_questions=None):
    """
    Load TriviaQA dataset from JSON format.
    
    Args:
        filepath: Path to JSON file
        max_knowledge: Maximum number of knowledge items to keep
        max_questions: Maximum number of questions to keep
        
    Returns:
        tuple: (list of document texts, list of question-answer pairs)
    """
    # Open and read the JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize lists
    text_list = []
    questions = []
    answers = []
    
    # Process each row in the dataset
    for row in data.get('rows', []):
        try:
            row_data = row.get('row', {})
            if not row_data:
                continue
                
            # Get the question
            question = row_data.get('question', '').strip()
            if not question:
                continue
                
            # Get the answer
            answer_data = row_data.get('answer', {})
            if isinstance(answer_data, str):
                try:
                    answer_data = json.loads(answer_data)
                except json.JSONDecodeError:
                    continue
                    
            answer = answer_data.get('value', '').strip()
            if not answer:
                continue
                
            # Get the search results as knowledge text
            search_results = row_data.get('search_results', {})
            if isinstance(search_results, str):
                try:
                    search_results = json.loads(search_results)
                except json.JSONDecodeError:
                    continue
            
            # Extract search contexts and descriptions
            contexts = search_results.get('search_context', [])
            descriptions = search_results.get('description', [])
            
            # Combine contexts and descriptions for knowledge text
            text_parts = []
            if contexts:
                text_parts.extend(contexts)
            if descriptions:
                text_parts.extend(descriptions)
                
            if not text_parts:
                continue
                
            # Join all text parts and clean
            text = '\n'.join(text_parts).strip()
            if not text:
                continue
                
            text_list.append(text)
            questions.append(question)
            answers.append(answer)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"Loaded {len(text_list)} documents with {len(questions)} questions from TriviaQA dataset")
    
    # Apply limits if specified
    if max_knowledge is not None:
        text_list = text_list[:max_knowledge]
    if max_questions is not None:
        questions = questions[:max_questions]
        answers = answers[:max_questions]
    
    print(f"After applying limits: {len(text_list)} documents with {len(questions)} questions")
    
    # Create dataset iterator
    dataset = list(zip(questions, answers))
    
    return text_list, dataset


def get(dataset_name, max_knowledge=None, max_questions=None):
    """
    Get dataset based on name.
    
    Args:
        dataset_name (str): Name of the dataset
        max_knowledge (int, optional): Maximum number of knowledge items
        max_questions (int, optional): Maximum number of questions
        
    Returns:
        tuple: (list of document texts, list of question-answer pairs)
    """
    if dataset_name == 'trivia-qa-train':
        filepath = './datasets/trivia_qa/trivia_qa_rc_train.json'
        return trivia_qa(filepath, max_knowledge, max_questions)
    elif dataset_name == 'natural-questions-train':
        texts, qa_pairs, _ = natural_questions(
            filepath="./datasets/googlenaturalquestions/v1.0-simplified_simplified-nq-train.jsonl.gz",
            max_knowledge=max_knowledge,
            max_question=max_questions,
            split=None,
            split_ratio=(0.8, 0.1, 0.1),
            random_seed=None
        )
        return texts, qa_pairs
    elif dataset_name == 'natural-questions-test':
        texts, qa_pairs, _ = natural_questions(
            filepath="./datasets/googlenaturalquestions/v1.0-simplified_simplified-nq-test.jsonl.gz",
            max_knowledge=max_knowledge,
            max_question=max_questions,
            split=None,
            split_ratio=(0.8, 0.1, 0.1),
            random_seed=None
        )
        return texts, qa_pairs
    elif dataset_name == 'natural-questions-eval':
        texts, qa_pairs, _ = natural_questions(
            filepath="./datasets/googlenaturalquestions/v1.0-simplified_simplified-nq-dev.jsonl.gz",
            max_knowledge=max_knowledge,
            max_question=max_questions,
            split=None,
            split_ratio=(0.8, 0.1, 0.1),
            random_seed=None
        )
        return texts, qa_pairs
    elif dataset_name == 'kis_sample':
        path = "./datasets/rag_sample_qas_from_kis.csv"
        return kis(path)
    elif dataset_name == 'kis':
        path = "./datasets/synthetic_knowledge_items.csv"
        return kis(path)
    elif dataset_name == 'squad-dev':
        path = "./datasets/squad/dev-v1.1.json"
        return squad(path, max_knowledge, None, max_questions)
    elif dataset_name == 'squad-train':
        path = "./datasets/squad/train-v1.1.json"
        return squad(path, max_knowledge, None, max_questions)
    elif dataset_name == 'hotpotqa-dev':
        path = "./datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
        return hotpotqa(path, max_knowledge)
    elif dataset_name == 'hotpotqa-test':
        path = "./datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
        return hotpotqa(path, max_knowledge)
    elif dataset_name == 'hotpotqa-train':
        path = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        return hotpotqa(path, max_knowledge)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
