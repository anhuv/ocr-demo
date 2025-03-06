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

# Initialize Mistral client with API key
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("Please set the MISTRAL_API_KEY environment variable.")
client = Mistral(api_key=api_key)

# Helper function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error encoding image: {str(e)}"

# OCR with PDF URL
def ocr_pdf_url(pdf_url):
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": pdf_url},
            include_image_base64=True
        )
        markdown = ocr_response.pages[0].markdown if ocr_response.pages else str(ocr_response)
        return markdown  # Return raw markdown for gr.Markdown to render
    except Exception as e:
        return f"**Error:** {str(e)}"

# OCR with Uploaded PDF
def ocr_uploaded_pdf(pdf_file):
    try:
        uploaded_pdf = client.files.upload(
            file={"file_name": pdf_file.name, "content": open(pdf_file.name, "rb")},
            purpose="ocr"
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id, expiry=3600)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url.url},
            include_image_base64=True
        )
        markdown = ocr_response.pages[0].markdown if ocr_response.pages else str(ocr_response)
        return markdown
    except Exception as e:
        return f"**Error:** {str(e)}"

# OCR with Image URL
def ocr_image_url(image_url):
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": image_url}
        )
        markdown = ocr_response.pages[0].markdown if ocr_response.pages else str(ocr_response)
        return markdown
    except Exception as e:
        return f"**Error:** {str(e)}"

# OCR with Uploaded Image
def ocr_uploaded_image(image_file):
    try:
        base64_image = encode_image(image_file.name)
        if "Error" in base64_image:
            return f"**Error:** {base64_image}"
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        )
        markdown = ocr_response.pages[0].markdown if ocr_response.pages else str(ocr_response)
        return markdown
    except Exception as e:
        return f"**Error:** {str(e)}"

# Document Understanding
def document_understanding(doc_url, question):
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "document_url", "document_url": doc_url}
            ]}
        ]
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )
        return chat_response.choices[0].message.content  # Plain text output
    except Exception as e:
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
    try:
        image_path = Path(image_file.name)
        encoded_image = encode_image(image_path)
        if "Error" in encoded_image:
            return f"**Error:** {encoded_image}"
        base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

        # OCR processing
        image_response = client.ocr.process(
            document={"type": "image_url", "image_url": base64_data_url},
            model="mistral-ocr-latest"
        )
        image_ocr_markdown = image_response.pages[0].markdown

        # Structured output with pixtral-12b-latest
        chat_response = client.chat.complete(
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
        
        response_dict = json.loads(chat_response.choices[0].message.content)
        structured_response = StructuredOCR.parse_obj({
            "file_name": image_path.name,
            "topics": response_dict.get("topics", []),
            "languages": [Language[l] for l in response_dict.get("languages", ["English"]) if l in languages.values()],
            "ocr_contents": response_dict.get("ocr_contents", {})
        })
        # Return as Markdown code block
        return f"```json\n{json.dumps(structured_response.dict(), indent=4)}\n```"
    except Exception as e:
        return f"**Error:** {str(e)}"

# Gradio Interface
with gr.Blocks(title="Mistral OCR & Structured Output App") as demo:
    gr.Markdown("# Mistral OCR & Structured Output App")
    gr.Markdown("Extract text from PDFs and images, ask questions about documents, or get structured JSON output in Markdown format!")

    with gr.Tab("OCR with PDF URL"):
        pdf_url_input = gr.Textbox(label="PDF URL", placeholder="e.g., https://arxiv.org/pdf/2201.04234")
        pdf_url_output = gr.Markdown(label="OCR Result (Markdown)")
        pdf_url_button = gr.Button("Process PDF")
        pdf_url_button.click(ocr_pdf_url, inputs=pdf_url_input, outputs=pdf_url_output)

    with gr.Tab("OCR with Uploaded PDF"):
        pdf_file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_file_output = gr.Markdown(label="OCR Result (Markdown)")
        pdf_file_button = gr.Button("Process Uploaded PDF")
        pdf_file_button.click(ocr_uploaded_pdf, inputs=pdf_file_input, outputs=pdf_file_output)

    with gr.Tab("OCR with Image URL"):
        image_url_input = gr.Textbox(label="Image URL", placeholder="e.g., https://example.com/image.jpg")
        image_url_output = gr.Markdown(label="OCR Result (Markdown)")
        image_url_button = gr.Button("Process Image")
        image_url_button.click(ocr_image_url, inputs=image_url_input, outputs=image_url_output)

    with gr.Tab("OCR with Uploaded Image"):
        image_file_input = gr.File(label="Upload Image", file_types=[".jpg", ".png"])
        image_file_output = gr.Markdown(label="OCR Result (Markdown)")
        image_file_button = gr.Button("Process Uploaded Image")
        image_file_button.click(ocr_uploaded_image, inputs=image_file_input, outputs=image_file_output)

    with gr.Tab("Document Understanding"):
        doc_url_input = gr.Textbox(label="Document URL", placeholder="e.g., https://arxiv.org/pdf/1805.04770")
        question_input = gr.Textbox(label="Question", placeholder="e.g., What is the last sentence?")
        doc_output = gr.Textbox(label="Answer")  # Keep as Textbox for plain text
        doc_button = gr.Button("Ask Question")
        doc_button.click(document_understanding, inputs=[doc_url_input, question_input], outputs=doc_output)

    with gr.Tab("Structured OCR"):
        struct_image_input = gr.File(label="Upload Image", file_types=[".jpg", ".png"])
        struct_output = gr.Markdown(label="Structured JSON Output (Markdown)")
        struct_button = gr.Button("Get Structured Output")
        struct_button.click(structured_ocr, inputs=struct_image_input, outputs=struct_output)

# Launch the app
demo.launch(share=True, debug=True)