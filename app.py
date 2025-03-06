import os
import base64
import gradio as gr
from mistralai import Mistral
from mistralai.models import OCRResponse
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
import pycountry
import json
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Mistral client with API key
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable is not set. Please configure it.")
client = Mistral(api_key=api_key)

# Helper function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return f"Error encoding image: {str(e)}"

# Retry-enabled API call helpers
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ocr_api(document):
    return client.ocr.process(model="mistral-ocr-latest", document=document)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_chat_complete(model, messages, **kwargs):
    return client.chat.complete(model=model, messages=messages, **kwargs)

# Helper function to get file content (handles both string paths and file-like objects)
def get_file_content(file_input):
    if isinstance(file_input, str):  # Gradio 3.x: file path
        with open(file_input, "rb") as f:
            return f.read()
    else:  # Gradio 4.x or file-like object
        return file_input.read()

# OCR with PDF URL
def ocr_pdf_url(pdf_url):
    logger.info(f"Processing PDF URL: {pdf_url}")
    try:
        ocr_response = call_ocr_api({"type": "document_url", "document_url": pdf_url})
        try:
            markdown = ocr_response.pages[0].markdown
        except (IndexError, AttributeError):
            markdown = "No text extracted or response invalid."
        logger.info("Successfully processed PDF URL")
        return markdown
    except Exception as e:
        logger.error(f"Error processing PDF URL: {str(e)}")
        return f"**Error:** {str(e)}"

# OCR with Uploaded PDF
def ocr_uploaded_pdf(pdf_file):
    logger.info(f"Processing uploaded PDF: {getattr(pdf_file, 'name', 'unknown')}")
    temp_path = None
    try:
        # Get file content (handles both string and file-like objects)
        content = get_file_content(pdf_file)
        # Use tempfile to handle uploaded file securely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        uploaded_pdf = client.files.upload(
            file={"file_name": temp_path, "content": open(temp_path, "rb")},
            purpose="ocr"
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id, expiry=7200)  # 2 hours
        ocr_response = call_ocr_api({"type": "document_url", "document_url": signed_url.url})
        try:
            markdown = ocr_response.pages[0].markdown
        except (IndexError, AttributeError):
            markdown = "No text extracted or response invalid."
        logger.info("Successfully processed uploaded PDF")
        return markdown
    except Exception as e:
        logger.error(f"Error processing uploaded PDF: {str(e)}")
        return f"**Error:** {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# OCR with Image URL
def ocr_image_url(image_url):
    logger.info(f"Processing image URL: {image_url}")
    try:
        ocr_response = call_ocr_api({"type": "image_url", "image_url": image_url})
        try:
            markdown = ocr_response.pages[0].markdown
        except (IndexError, AttributeError):
            markdown = "No text extracted or response invalid."
        logger.info("Successfully processed image URL")
        return markdown
    except Exception as e:
        logger.error(f"Error processing image URL: {str(e)}")
        return f"**Error:** {str(e)}"

# OCR with Uploaded Image
def ocr_uploaded_image(image_file):
    logger.info(f"Processing uploaded image: {getattr(image_file, 'name', 'unknown')}")
    temp_path = None
    try:
        # Get file content (handles both string and file-like objects)
        content = get_file_content(image_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        encoded_image = encode_image(temp_path)
        if "Error" in encoded_image:
            raise ValueError(encoded_image)
        base64_data_url = f"data:image/jpeg;base64,{encoded_image}"
        ocr_response = call_ocr_api({"type": "image_url", "image_url": base64_data_url})
        try:
            markdown = ocr_response.pages[0].markdown
        except (IndexError, AttributeError):
            markdown = "No text extracted or response invalid."
        logger.info("Successfully processed uploaded image")
        return markdown
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        return f"**Error:** {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Document Understanding
def document_understanding(doc_url, question):
    logger.info(f"Processing document understanding - URL: {doc_url}, Question: {question}")
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "document_url", "document_url": doc_url}
            ]}
        ]
        chat_response = call_chat_complete(model="mistral-small-latest", messages=messages)
        try:
            content = chat_response.choices[0].message.content
        except (IndexError, AttributeError):
            content = "No response received from the API."
        logger.info("Successfully processed document understanding")
        return content
    except Exception as e:
        logger.error(f"Error in document understanding: {str(e)}")
        return f"**Error:** {str(e)}"

# Structured OCR Setup
languages = {lang.alpha_2: lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')}

class LanguageMeta(Enum.__class__):
    def __new__(metacls, cls, bases, classdict):
        for code, name in languages.items():
            classdict[name.upper().replace(' ', '_')] = name
        return super().__new__(metacls, cls, bases, classdict)

class Language(Enum, metaclass=LanguageMeta):
    pass

class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: list[Language]
    ocr_contents: dict

def structured_ocr(image_file):
    logger.info(f"Processing structured OCR for image: {getattr(image_file, 'name', 'unknown')}")
    temp_path = None
    try:
        # Get file content (handles both string and file-like objects)
        content = get_file_content(image_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        image_path = Path(temp_path)
        encoded_image = encode_image(temp_path)
        if "Error" in encoded_image:
            raise ValueError(encoded_image)
        base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

        image_response = call_ocr_api({"type": "image_url", "image_url": base64_data_url})
        try:
            image_ocr_markdown = image_response.pages[0].markdown
        except (IndexError, AttributeError):
            image_ocr_markdown = "No text extracted."

        chat_response = call_chat_complete(
            model="pixtral-12b-latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": base64_data_url},
                    {"type": "text", "text": (
                        f"This is the image's OCR in markdown:\n<BEGIN_IMAGE_OCR>\n{image_ocr_markdown}\n<END_IMAGE_OCR>.\n"
                        "Convert this into a structured JSON response with the OCR contents in a sensible dictionary."
                    )}
                ],
            }],
            response_format={"type": "json_object"},
            temperature=0
        )

        try:
            content = chat_response.choices[0].message.content
            response_dict = json.loads(content)
        except (json.JSONDecodeError, IndexError, AttributeError):
            logger.error("Failed to parse structured response")
            return "Failed to parse structured response. Please try again."

        language_members = {member.value: member for member in Language}
        valid_languages = [l for l in response_dict.get("languages", ["English"]) if l in language_members]
        languages = [language_members[l] for l in valid_languages] if valid_languages else [Language.ENGLISH]

        structured_response = StructuredOCR(
            file_name=image_path.name,
            topics=response_dict.get("topics", []),
            languages=languages,
            ocr_contents=response_dict.get("ocr_contents", {})
        )
        logger.info("Successfully processed structured OCR")
        return f"```json\n{json.dumps(structured_response.dict(), indent=4)}\n```"
    except Exception as e:
        logger.error(f"Error processing structured OCR: {str(e)}")
        return f"**Error:** {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Gradio Interface
with gr.Blocks(title="Mistral OCR & Structured Output App") as demo:
    gr.Markdown("# Mistral OCR & Structured Output App")
    gr.Markdown("Extract text from PDFs and images, ask questions about documents, or get structured JSON output!")

    with gr.Tab("OCR with PDF URL"):
        pdf_url_input = gr.Textbox(label="PDF URL", placeholder="e.g., https://arxiv.org/pdf/2201.04234")
        pdf_url_output = gr.Textbox(label="OCR Result (Markdown)")
        pdf_url_button = gr.Button("Process PDF")
        pdf_url_button.click(ocr_pdf_url, inputs=pdf_url_input, outputs=pdf_url_output)

    with gr.Tab("OCR with Uploaded PDF"):
        pdf_file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_file_output = gr.Textbox(label="OCR Result (Markdown)")
        pdf_file_button = gr.Button("Process Uploaded PDF")
        pdf_file_button.click(ocr_uploaded_pdf, inputs=pdf_file_input, outputs=pdf_file_output)

    with gr.Tab("OCR with Image URL"):
        image_url_input = gr.Textbox(label="Image URL", placeholder="e.g., https://example.com/image.jpg")
        image_url_output = gr.Textbox(label="OCR Result (Markdown)")
        image_url_button = gr.Button("Process Image")
        image_url_button.click(ocr_image_url, inputs=image_url_input, outputs=image_url_output)

    with gr.Tab("OCR with Uploaded Image"):
        image_file_input = gr.File(label="Upload Image", file_types=[".jpg", ".png"])
        image_file_output = gr.Textbox(label="OCR Result (Markdown)")
        image_file_button = gr.Button("Process Uploaded Image")
        image_file_button.click(ocr_uploaded_image, inputs=image_file_input, outputs=image_file_output)

    with gr.Tab("Document Understanding"):
        doc_url_input = gr.Textbox(label="Document URL", placeholder="e.g., https://arxiv.org/pdf/1805.04770")
        question_input = gr.Textbox(label="Question", placeholder="e.g., What is the last sentence?")
        doc_output = gr.Textbox(label="Answer")
        doc_button = gr.Button("Ask Question")
        doc_button.click(document_understanding, inputs=[doc_url_input, question_input], outputs=doc_output)

    with gr.Tab("Structured OCR"):
        struct_image_input = gr.File(label="Upload Image", file_types=[".jpg", ".png"])
        struct_output = gr.Textbox(label="Structured JSON Output")
        struct_button = gr.Button("Get Structured Output")
        struct_button.click(structured_ocr, inputs=struct_image_input, outputs=struct_output)

demo.launch(share=True, debug=True)