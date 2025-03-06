import os
import base64
import gradio as gr
from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai.exceptions import MistralException  # Import Mistral-specific exceptions
from pathlib import Path
from pydantic import BaseModel
import pycountry
import json
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import tempfile
from typing import Union, Dict, List
from contextlib import contextmanager

# Constants
DEFAULT_LANGUAGE = "English"
SUPPORTED_IMAGE_TYPES = [".jpg", ".png"]
SUPPORTED_PDF_TYPES = [".pdf"]
TEMP_FILE_EXPIRY = 7200  # 2 hours in seconds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key must be provided")
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        # Test API key validity on initialization
        try:
            self.client.models.list()  # Simple API call to validate key
        except MistralException as e:
            raise ValueError(f"Invalid API key: {str(e)}")

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    @contextmanager
    def _temp_file(content: bytes, suffix: str) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            temp_file.write(content)
            temp_file.close()
            yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry_if_exception_type(MistralException))
    def _call_ocr_api(self, document: Dict) -> OCRResponse:
        try:
            return self.client.ocr.process(model="mistral-ocr-latest", document=document)
        except MistralException as e:
            logger.error(f"OCR API call failed: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry_if_exception_type(MistralException))
    def _call_chat_complete(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        try:
            return self.client.chat.complete(model=model, messages=messages, **kwargs)
        except MistralException as e:
            logger.error(f"Chat complete API call failed: {str(e)}")
            raise

    def _get_file_content(self, file_input: Union[str, bytes]) -> bytes:
        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                return f.read()
        return file_input.read() if hasattr(file_input, 'read') else file_input

    def ocr_pdf_url(self, pdf_url: str) -> str:
        logger.info(f"Processing PDF URL: {pdf_url}")
        try:
            response = self._call_ocr_api({"type": "document_url", "document_url": pdf_url})
            return self._extract_markdown(response)
        except Exception as e:
            return self._handle_error("PDF URL processing", e)

    def ocr_uploaded_pdf(self, pdf_file: Union[str, bytes]) -> str:
        file_name = getattr(pdf_file, 'name', 'unknown')
        logger.info(f"Processing uploaded PDF: {file_name}")
        try:
            content = self._get_file_content(pdf_file)
            with self._temp_file(content, ".pdf") as temp_path:
                uploaded_file = self.client.files.upload(
                    file={"file_name": temp_path, "content": open(temp_path, "rb")},
                    purpose="ocr"
                )
                signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=TEMP_FILE_EXPIRY)
                response = self._call_ocr_api({"type": "document_url", "document_url": signed_url.url})
                return self._extract_markdown(response)
        except Exception as e:
            return self._handle_error("uploaded PDF processing", e)

    def ocr_image_url(self, image_url: str) -> str:
        logger.info(f"Processing image URL: {image_url}")
        try:
            response = self._call_ocr_api({"type": "image_url", "image_url": image_url})
            return self._extract_markdown(response)
        except Exception as e:
            return self._handle_error("image URL processing", e)

    def ocr_uploaded_image(self, image_file: Union[str, bytes]) -> str:
        file_name = getattr(image_file, 'name', 'unknown')
        logger.info(f"Processing uploaded image: {file_name}")
        try:
            content = self._get_file_content(image_file)
            with self._temp_file(content, ".jpg") as temp_path:
                encoded_image = self._encode_image(temp_path)
                base64_url = f"data:image/jpeg;base64,{encoded_image}"
                response = self._call_ocr_api({"type": "image_url", "image_url": base64_url})
                return self._extract_markdown(response)
        except Exception as e:
            return self._handle_error("uploaded image processing", e)

    def document_understanding(self, doc_url: str, question: str) -> str:
        logger.info(f"Document understanding - URL: {doc_url}, Question: {question}")
        try:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "document_url", "document_url": doc_url}
            ]}]
            response = self._call_chat_complete(model="mistral-small-latest", messages=messages)
            return response.choices[0].message.content if response.choices else "No response received"
        except Exception as e:
            return self._handle_error("document understanding", e)

    def structured_ocr(self, image_file: Union[str, bytes]) -> str:
        file_name = getattr(image_file, 'name', 'unknown')
        logger.info(f"Processing structured OCR for: {file_name}")
        try:
            content = self._get_file_content(image_file)
            with self._temp_file(content, ".jpg") as temp_path:
                encoded_image = self._encode_image(temp_path)
                base64_url = f"data:image/jpeg;base64,{encoded_image}"
                ocr_response = self._call_ocr_api({"type": "image_url", "image_url": base64_url})
                markdown = self._extract_markdown(ocr_response)

                chat_response = self._call_chat_complete(
                    model="pixtral-12b-latest",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": base64_url},
                            {"type": "text", "text": (
                                f"OCR result:\n<BEGIN_IMAGE_OCR>\n{markdown}\n<END_IMAGE_OCR>\n"
                                "Convert to structured JSON with file_name, topics, languages, and ocr_contents"
                            )}
                        ]
                    }],
                    response_format={"type": "json_object"},
                    temperature=0
                )

                content = json.loads(chat_response.choices[0].message.content if chat_response.choices else "{}")
                return self._format_structured_response(temp_path, content)
        except Exception as e:
            return self._handle_error("structured OCR", e)

    @staticmethod
    def _extract_markdown(response: OCRResponse) -> str:
        return response.pages[0].markdown if response.pages else "No text extracted"

    @staticmethod
    def _handle_error(context: str, error: Exception) -> str:
        logger.error(f"Error in {context}: {str(error)}")
        return f"**Error:** {str(error)}"

    @staticmethod
    def _format_structured_response(file_path: str, content: Dict) -> str:
        languages = {lang.alpha_2: lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')}
        valid_langs = [l for l in content.get("languages", [DEFAULT_LANGUAGE]) if l in languages.values()]
        
        response = {
            "file_name": Path(file_path).name,
            "topics": content.get("topics", []),
            "languages": valid_langs or [DEFAULT_LANGUAGE],
            "ocr_contents": content.get("ocr_contents", {})
        }
        return f"```json\n{json.dumps(response, indent=4)}\n```"

def create_interface():
    with gr.Blocks(title="Mistral OCR & Structured Output App") as demo:
        gr.Markdown("# Mistral OCR & Structured Output App")
        gr.Markdown("Enter your Mistral API key below to use the app. Extract text from PDFs and images or get structured JSON output.")

        # API Key input
        api_key_input = gr.Textbox(
            label="Mistral API Key",
            placeholder="Enter your Mistral API key here",
            type="password"
        )
        
        # Function to initialize processor with API key
        def initialize_processor(api_key):
            try:
                processor = OCRProcessor(api_key)
                return processor, "**Success:** API key set and validated!"
            except ValueError as e:
                return None, f"**Error:** {str(e)}"
            except Exception as e:
                return None, f"**Error:** Unexpected error: {str(e)}"

        # Store processor state
        processor_state = gr.State()
        api_status = gr.Markdown("API key not set. Please enter and set your key.")

        # Button to set API key
        set_api_button = gr.Button("Set API Key")
        set_api_button.click(
            fn=initialize_processor,
            inputs=api_key_input,
            outputs=[processor_state, api_status]
        )

        tabs = [
            ("OCR with PDF URL", gr.Textbox, "ocr_pdf_url", "PDF URL", None),
            ("OCR with Uploaded PDF", gr.File, "ocr_uploaded_pdf", "Upload PDF", SUPPORTED_PDF_TYPES),
            ("OCR with Image URL", gr.Textbox, "ocr_image_url", "Image URL", None),
            ("OCR with Uploaded Image", gr.File, "ocr_uploaded_image", "Upload Image", SUPPORTED_IMAGE_TYPES),
            ("Structured OCR", gr.File, "structured_ocr", "Upload Image", SUPPORTED_IMAGE_TYPES),
        ]

        for name, input_type, fn_name, label, file_types in tabs:
            with gr.Tab(name):
                if input_type == gr.Textbox:
                    inputs = input_type(label=label, placeholder=f"e.g., https://example.com/{label.lower().replace(' ', '')}")
                else:  # gr.File
                    inputs = input_type(label=label, file_types=file_types)
                output = gr.Markdown(label="Result")
                button_label = name.replace("OCR with ", "").replace("Structured ", "Get Structured ")

                def process_with_api(processor, input_data):
                    if not processor:
                        return "**Error:** Please set a valid API key first."
                    fn = getattr(processor, fn_name)
                    return fn(input_data)

                gr.Button(f"Process {button_label}").click(
                    fn=process_with_api,
                    inputs=[processor_state, inputs],
                    outputs=output
                )

        with gr.Tab("Document Understanding"):
            doc_url = gr.Textbox(label="Document URL", placeholder="e.g., https://arxiv.org/pdf/1805.04770")
            question = gr.Textbox(label="Question", placeholder="e.g., What is the last sentence?")
            output = gr.Markdown(label="Answer")

            def doc_understanding_with_api(processor, url, q):
                if not processor:
                    return "**Error:** Please set a valid API key first."
                return processor.document_understanding(url, q)

            gr.Button("Ask Question").click(
                fn=doc_understanding_with_api,
                inputs=[processor_state, doc_url, question],
                outputs=output
            )

    return demo

if __name__ == "__main__":
    create_interface().launch(share=True, debug=True)