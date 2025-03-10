import os
import re
import json
import time
import sys
import logging
from typing import List, Dict, Any

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import PyPDF2
import markdown
from openai import OpenAI

# ==============================
# Configuration & Logging Setup
# ==============================
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-2.0-flash-exp:free"
DEFAULT_TEMPERATURE = 0.3

# Directories & file parameters
PDF_DOWNLOAD_DIR = "./arxiv_pdfs"
WEB_DATA_DIR = "./web_data"
MARKDOWN_DATA_DIR = "./data"
OUTPUT_CHUNKS_FILE = "document_chunks.json"
OUTPUT_QA_FILE = "alpaca_qa_dataset.json"

# Chunking parameters (in words)
CHUNK_WORD_COUNT = 1000

# QA generation parameters
QUESTIONS_PER_CHUNK = 3
QUESTION_MAX_TOKENS = 150
ANSWER_MAX_TOKENS = 400
API_DELAY = 60  # seconds delay to avoid rate limits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def generate_text_openrouter(prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generate text using the OpenRouter API via the OpenAI client.
    Includes detailed error handling for various scenarios.
    """
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    try:
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional; update if needed.
                "X-Title": "<YOUR_SITE_NAME>",       # Optional; update if needed.
            },
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=max_new_tokens
        )
        # print(f"respocse \n {response}")
        # Check if response is valid and has choices.
        if not response or not hasattr(response, "choices") or not response.choices:
            logger.error("Invalid response format received: %s", response)
            return "No valid response received from the API."

        # print(response.choices[0].message.content)
        answer = response.choices[0].message.content
        logger.info("Generated response: %s", answer)
        # print(f"Generated Response {answer}")
        return answer

    except Exception as e:
        error_message = str(e).lower()
        if "rate limit" in error_message:
            logger.error("Rate limit error: %s", e)
            return "API rate limit exceeded. Please try again later."
        elif "timeout" in error_message:
            logger.error("Timeout error: %s", e)
            return "API request timed out. Please try again."
        elif "authentication" in error_message or "invalid api key" in error_message:
            logger.error("Authentication error: %s", e)
            return "Authentication failed. Please check your API key."
        else:
            logger.error("Unexpected error with OpenRouter API: %s", e)
            return f"An unexpected error occurred: {e}"


def generate_questions(context: str, num_questions: int = 3) -> List[str]:
    prompt = f"""You are an expert academic researcher specialized in generating insightful questions for educational content.

CONTEXT:
{context}

Based on this context, generate {num_questions} intellectually stimulating questions that:
1. Require basic understanding of the technical concepts 
2. Include a mix of factual, conceptual, and analytical questions
3. Target multi-level complexity (foundational → expert)
4. Target different cognitive levels (knowledge, application, analysis, evaluation)
5. Add simple to complex indepth questions

Format your response as a numbered list (1., 2., 3.) with ONLY the questions.
"""
    generated = generate_text_openrouter(prompt, max_new_tokens=QUESTION_MAX_TOKENS)
    pattern = r'^\d+\.\s+(.+)$'
    questions = [match.group(1).strip() for match in re.finditer(pattern, generated, re.MULTILINE)]
    logger.info("Extracted questions: %s", questions)
    return questions[:num_questions]

def generate_answer(context: str, question: str) -> str:
    prompt = f"""You are an AI research expert answering questions about technical AI literature.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Provide an authoritative answer that:
    1. Thoroughly explains the AI concepts, methodologies, and implications
    2. References specific details from the research paper or technical document
    3. Uses precise AI terminology while maintaining clarity
    4. Organizes information with a logical structure (definitions, explanations, examples)
    5. Balances technical depth with accessibility (200-300 words)

    Your answer:"""

    answer = generate_text_openrouter(prompt, max_new_tokens=ANSWER_MAX_TOKENS)
    logger.info("Generated answer for question: %s", question)
    return answer.strip()

# ==============================
# Document Processing Functions
# ==============================
def download_arxiv_pdf(arxiv_id: str, output_dir: str = PDF_DOWNLOAD_DIR) -> str:
    """
    Download a PDF from arXiv given its identifier.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    try:
        logger.info("Downloading PDF from: %s", pdf_url)
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded PDF to: %s", output_path)
        return output_path
    except Exception as e:
        logger.error("Error downloading PDF for %s: %s", arxiv_id, str(e))
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logger.info("Extracted text from PDF: %s", pdf_path)
    except Exception as e:
        logger.error("Error extracting text from %s: %s", pdf_path, str(e))
    return text

def extract_text_from_web(url: str) -> str:
    """
    Download a webpage and extract its plain text using BeautifulSoup.
    """
    try:
        logger.info("Downloading webpage: %s", url)
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        logger.info("Extracted text from webpage: %s", url)
        return text
    except Exception as e:
        logger.error("Error downloading/extracting webpage %s: %s", url, str(e))
        return ""

def chunk_text(text: str, chunk_word_count: int = CHUNK_WORD_COUNT) -> List[str]:
    """
    Split text into chunks of approximately `chunk_word_count` words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_word_count):
        chunk = " ".join(words[i:i+chunk_word_count])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def process_arxiv_pdfs(arxiv_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Download PDFs for given arXiv IDs, extract text, and chunk it.
    """
    chunks = []
    for arxiv_id in tqdm(arxiv_ids, desc="Processing arXiv PDFs"):
        pdf_path = download_arxiv_pdf(arxiv_id)
        if not pdf_path:
            continue
        pdf_text = extract_text_from_pdf(pdf_path)
        for idx, chunk in enumerate(chunk_text(pdf_text)):
            chunks.append({
                "content": chunk,
                "source": f"arXiv:{arxiv_id}",
                "chunk_index": idx,
                "type": "pdf"
            })
    return chunks

def process_web_pages(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Download webpages, extract text, and chunk it.
    """
    chunks = []
    for url in tqdm(urls, desc="Processing Web Pages"):
        web_text = extract_text_from_web(url)
        for idx, chunk in enumerate(chunk_text(web_text)):
            chunks.append({
                "content": chunk,
                "source": url,
                "chunk_index": idx,
                "type": "web"
            })
    return chunks

def process_markdown_files(md_dir: str) -> List[Dict[str, Any]]:
    """
    Process Markdown files in a directory, extract text, and chunk it.
    """
    chunks = []
    if os.path.exists(md_dir):
        for filename in tqdm(os.listdir(md_dir), desc="Processing Markdown files"):
            if filename.lower().endswith(('.md', '.markdown')):
                md_path = os.path.join(md_dir, filename)
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()
                    metadata = {}
                    metadata_match = re.search(r'^---\s*([\s\S]*?)\s*---', md_text)
                    if metadata_match:
                        metadata_text = metadata_match.group(1)
                        md_text = md_text[metadata_match.end():].strip()
                        for line in metadata_text.split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                metadata[key.strip()] = value.strip()
                    html = markdown.markdown(md_text)
                    soup = BeautifulSoup(html, 'html.parser')
                    plain_text = soup.get_text()
                    words = plain_text.split()
                    for i in range(0, len(words), 1000):
                        chunk_text = ' '.join(words[i:i + 1000])
                        if chunk_text.strip():
                            chunks.append({
                                "content": chunk_text.strip(),
                                "source": md_path,
                                "metadata": metadata,
                                "type": "markdown"
                            })
                    print(f"Extracted {len(chunks)} chunks from Markdown: {md_path}")
                except Exception as e:
                    print(f"Error processing Markdown {md_path}: {str(e)}")
    else:
        print(f"Markdown directory {md_dir} does not exist.")
    return chunks

# ==============================
# QA Generation & Dataset Creation
# ==============================
def format_to_alpaca(question_answer_pairs: List[Dict]) -> List[Dict]:
    """
    Convert QA pairs into Alpaca fine-tuning format.
    """
    alpaca_data = []
    for pair in question_answer_pairs:
        alpaca_data.append({
            "instruction": pair["question"],
            "input": "",
            "output": pair["answer"],
            "source": pair["source"],
            "metadata": pair.get("metadata", {})
        })
    return alpaca_data

def generate_qa_dataset(chunks: List[Dict[str, Any]], questions_per_chunk: int = QUESTIONS_PER_CHUNK) -> List[Dict]:
    """
    For each text chunk, generate QA pairs using the LLM.
    """
    qa_pairs = []
    for chunk in tqdm(chunks, desc="Generating QA pairs"):
        context = chunk["content"]
        questions = generate_questions(context, questions_per_chunk)
        for question in questions:
            answer = generate_answer(context, question)
            qa_pair = {
                "question": question,
                "answer": answer,
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "type": chunk["type"]
            }
            qa_pairs.append(qa_pair)
            logger.info("Generated QA pair: %s", qa_pair)
        time.sleep(API_DELAY)
    return qa_pairs

def save_dataset(dataset: List[Dict], output_file: str):
    """
    Save dataset (list of dictionaries) to a JSON file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info("Dataset saved to %s", output_file)
    except Exception as e:
        logger.error("Error saving dataset: %s", str(e))


import re
import html

def clean_text_chunk(text: str) -> str:
    """
    Clean a text chunk by:
    - Converting HTML entities to plain text.
    - Removing extra whitespace and newlines.
    - Removing unwanted special characters.
    - Checking for error indicators ("404 error", "not found", etc.).
    
    If any error indicators are found, an empty string is returned to indicate that
    the chunk should be discarded.
    """
    # Convert HTML entities to plain text.
    text = html.unescape(text)
    
    # Remove unwanted special characters; keep alphanumerics, spaces, and selected punctuation.
    text = re.sub(r'[^\w\s\.,;:?!-]', '', text)
    
    # Normalize whitespace: replace multiple spaces/newlines with a single space.
    text = re.sub(r'\s+', ' ', text)
    
    # Trim leading/trailing whitespace.
    text = text.strip()
    
    # Define common error indicators.
    error_indicators = ["404 error", "not found", "page not found", "error 404"]
    for indicator in error_indicators:
        if indicator.lower() in text.lower():
            # Return empty string to signal that the entire chunk should be discarded.
            return ""
    
    return text

def process_batches(chunks: List[Dict[str, Any]], batch_size: int = 10) -> List[List[Dict]]:
    """
    Split the list of chunks into batches.
    """
    return [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]

# ==============================
# Main Execution
# ==============================
def main():
     # Example arXiv IDs and web URLs; adjust these lists as needed.
    arxiv_ids = [
        '2405.09939v1', '2501.12948v1','2503.04725v1', '2503.04723v1', '2503.04722v1',
                 '2503.04715v1', '2503.04713v1', '2503.04710v1', '2503.04704v1', '2503.04697v1',
                 '2503.04680v1', '2503.04679v1', '2503.04647v1', '2503.04641v1', '2503.04636v1',
                 '2503.04626v1', '2503.04606v1', '2503.04598v1', '2503.04596v1', '2503.04569v1',
                 '2503.04564v1', '2503.04556v1'
                 ]

    web_urls = [
            "https://www.anthropic.com/news/model-context-protocol",
            "https://www.anthropic.com/news/contextual-retrieval",
            # Documents & Reports
            "https://www.mckinsey.com/capabilities/quantumblack/our-insights/ai-in-the-workplace-a-report-for-2025",  # AI in the Workplace: A Report for 2025
            "https://www.oecd.org/digital/artificial-intelligence/statement-on-inclusive-and-sustainable-artificial-intelligence-2025.pdf",  # Statement on Inclusive and Sustainable Artificial Intelligence
            "https://carnegieendowment.org/research/2025/02/the-missing-pieces-in-indias-ai-puzzle-talent-data-and-randd",  # The Missing Pieces in India's AI Puzzle: Talent, Data, and R&D
            "https://www.theguardian.com/technology/2025/mar/01/bbc-ai-future-journalism-report-2025",  # BBC's AI-Powered Future: Journalism Reimagined
            "https://www.currentai.org/strategic-plan-2025",  # Current AI Strategic Plan 2025
            "https://aiindex.stanford.edu/wp-content/uploads/2024/05/HAI_AI-Index-Report-2024.pdf",  # Stanford AI Index Report 2024
            "https://aiindex.stanford.edu/report/",  # AI Index Report 2024 – Interactive version
            "https://www.weka.io/resources/analyst-report/2024-global-trends-in-ai/",  # 2024 Global Trends in AI
            "https://insight7.io/best-ai-document-analysis-software-in-2024/",  # Best AI document analysis software in 2024
            "https://www.linkedin.com/pulse/top-20-best-ai-tools-2024-asterdio-gunjf",  # Top 20 Best AI Tools in 2024
    ]

    # Process chunks in batches to avoid losing all progress if an error occurs.
    batches = process_batches(all_chunks, batch_size=10)
    all_qa_pairs = []
    for batch_index, batch in enumerate(batches, start=6):
        logger.info("Processing batch %d/%d with %d chunks", batch_index, len(batches), len(batch))
        try:
            batch_qa_pairs = generate_qa_dataset(batch, questions_per_chunk=QUESTIONS_PER_CHUNK)
            all_qa_pairs.extend(batch_qa_pairs)
            # Save batch QA dataset separately.
            batch_output_file = f"alpaca_qa_dataset_batch_{batch_index}.json"
            alpaca_batch_dataset = format_to_alpaca(batch_qa_pairs)
            save_dataset(alpaca_batch_dataset, batch_output_file)
            logger.info("Batch %d saved with %d QA pairs.", batch_index, len(alpaca_batch_dataset))
        except Exception as batch_error:
            logger.error("Error processing batch %d: %s", batch_index, batch_error)
        # Optional: pause between batches to avoid rate limits.
        time.sleep(API_DELAY)

    # Optionally, save the combined QA dataset.
    alpaca_dataset = format_to_alpaca(all_qa_pairs)
    save_dataset(alpaca_dataset, OUTPUT_QA_FILE)
    logger.info("QA dataset generation complete! Total examples: %d", len(alpaca_dataset))

if __name__ == "__main__":
    main()
