import os
import base64
import gradio as gr
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from pathlib import Path
import pycountry
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import tempfile
from typing import Union, Dict, List, Optional, Tuple
from contextlib import contextmanager
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor
import time

# Constants
DEFAULT_LANGUAGE = "English"
SUPPORTED_IMAGE_TYPES = [".jpg", ".png", ".jpeg"]
SUPPORTED_PDF_TYPES = [".pdf"]
TEMP_FILE_EXPIRY = 7200  # 2 hours in seconds
UPLOAD_FOLDER = "./uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PDF_PAGES = 50

# Configuration
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, api_key: str):
        self.api_key = self._validate_api_key(api_key)
        self.client = Mistral(api_key=self.api_key)
        self._validate_client()

    @staticmethod
    def _validate_api_key(api_key: str) -> str:
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid API key must be provided")
        return api_key

    def _validate_client(self) -> None:
        try:
            models = self.client.models.list()
            if not models:
                raise ValueError("No models available")
        except Exception as e:
            raise ValueError(f"API key validation failed: {str(e)}")

    @staticmethod
    def _check_file_size(file_input: Union[str, bytes]) -> None:
        if isinstance(file_input, str) and os.path.exists(file_input):
            size = os.path.getsize(file_input)
        elif hasattr(file_input, 'read'):
            size = len(file_input.read())
            file_input.seek(0)  # Reset file pointer
        else:
            size = len(file_input)
        if size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")

    @staticmethod
    def _encode_image(image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            return None

    @staticmethod
    def _save_uploaded_file(file_input: Union[str, bytes], filename: str) -> str:
        clean_filename = os.path.basename(filename).replace(os.sep, "_")
        file_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{clean_filename}")
        
        try:
            if isinstance(file_input, str) and file_input.startswith("http"):
                response = requests.get(file_input, timeout=10)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            elif isinstance(file_input, str) and os.path.exists(file_input):
                shutil.copy2(file_input, file_path)
            else:
                with open(file_path, 'wb') as f:
                    if hasattr(file_input, 'read'):
                        shutil.copyfileobj(file_input, f)
                    else:
                        f.write(file_input)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Failed to save file at {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            raise

    @staticmethod
    def _pdf_to_images(pdf_path: str) -> List[str]:
        try:
            pdf_document = fitz.open(pdf_path)
            if pdf_document.page_count > MAX_PDF_PAGES:
                pdf_document.close()
                raise ValueError(f"PDF exceeds maximum page limit of {MAX_PDF_PAGES}")
            
            with ThreadPoolExecutor() as executor:
                image_paths = list(executor.map(
                    lambda i: OCRProcessor._convert_page(pdf_path, i),
                    range(pdf_document.page_count)
                ))
            pdf_document.close()
            return [path for path in image_paths if path]
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []

    @staticmethod
    def _convert_page(pdf_path: str, page_num: int) -> Optional[str]:
        try:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document[page_num]
            pix = page.get_pixmap(dpi=150)
            image_path = os.path.join(UPLOAD_FOLDER, f"page_{page_num + 1}_{int(time.time())}.png")
            pix.save(image_path)
            pdf_document.close()
            return image_path
        except Exception as e:
            logger.error(f"Error converting page {page_num}: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_ocr_api(self, document: Union[DocumentURLChunk, ImageURLChunk]) -> OCRResponse:
        return self.client.ocr.process(
            model="mistral-ocr-latest",
            document=document,
            include_image_base64=True
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_chat_complete(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        return self.client.chat.complete(model=model, messages=messages, **kwargs)

    def ocr_uploaded_pdf(self, pdf_file: Union[str, bytes]) -> Tuple[str, List[str]]:
        file_name = getattr(pdf_file, 'name', f"pdf_{int(time.time())}.pdf")
        logger.info(f"Processing uploaded PDF: {file_name}")
        try:
            self._check_file_size(pdf_file)
            pdf_path = self._save_uploaded_file(pdf_file, file_name)
            logger.info(f"Saved PDF to: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Saved PDF not found at: {pdf_path}")
            
            image_paths = self._pdf_to_images(pdf_path)
            
            with open(pdf_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={"file_name": file_name, "content": f},
                    purpose="ocr"
                )
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=TEMP_FILE_EXPIRY)
            response = self._call_ocr_api(DocumentURLChunk(document_url=signed_url.url))
            return self._get_combined_markdown(response), image_paths
        except Exception as e:
            return self._handle_error("PDF processing", e), []

    def ocr_uploaded_image(self, image_file: Union[str, bytes]) -> Tuple[str, str]:
        file_name = getattr(image_file, 'name', f"image_{int(time.time())}.jpg")
        logger.info(f"Processing uploaded image: {file_name}")
        try:
            self._check_file_size(image_file)
            image_path = self._save_uploaded_file(image_file, file_name)
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                raise ValueError("Failed to encode image")
            base64_url = f"data:image/jpeg;base64,{encoded_image}"
            response = self._call_ocr_api(ImageURLChunk(image_url=base64_url))
            return self._get_combined_markdown(response), image_path
        except Exception as e:
            return self._handle_error("image processing", e), None

    def document_understanding(self, doc_url: str, question: str) -> str:
        try:
            messages = [{"role": "user", "content": [
                TextChunk(text=question),
                DocumentURLChunk(document_url=doc_url)
            ]}]
            response = self._call_chat_complete(
                model="mistral-small-latest",
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._handle_error("document understanding", e)

    def structured_ocr(self, image_file: Union[str, bytes]) -> Tuple[str, str]:
        file_name = getattr(image_file, 'name', f"image_{int(time.time())}.jpg")
        try:
            self._check_file_size(image_file)
            image_path = self._save_uploaded_file(image_file, file_name)
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                raise ValueError("Failed to encode image")
            base64_url = f"data:image/jpeg;base64,{encoded_image}"
            
            ocr_response = self._call_ocr_api(ImageURLChunk(image_url=base64_url))
            markdown = self._get_combined_markdown(ocr_response)

            chat_response = self._call_chat_complete(
                model="pixtral-12b-latest",
                messages=[{
                    "role": "user",
                    "content": [
                        ImageURLChunk(image_url=base64_url),
                        TextChunk(text=(
                            f"This is image's OCR in markdown:\n<BEGIN_IMAGE_OCR>\n{markdown}\n<END_IMAGE_OCR>.\n"
                            "Convert this into a structured JSON response with file_name, topics, languages, and ocr_contents fields"
                        ))
                    ]
                }],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return self._format_structured_response(image_path, json.loads(chat_response.choices[0].message.content)), image_path
        except Exception as e:
            return self._handle_error("structured OCR", e), None

    @staticmethod
    def _get_combined_markdown(response: OCRResponse) -> str:
        return "\n\n".join(
            page.markdown for page in response.pages
            if page.markdown.strip()
        ) or "No text detected"

    @staticmethod
    def _handle_error(context: str, error: Exception) -> str:
        logger.error(f"Error in {context}: {str(error)}")
        return f"**Error in {context}:** {str(error)}"

    @staticmethod
    def _format_structured_response(file_path: str, content: Dict) -> str:
        languages = {lang.alpha_2: lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')}
        content_languages = content.get("languages", [DEFAULT_LANGUAGE])
        valid_langs = [l for l in content_languages if l in languages.values()] or [DEFAULT_LANGUAGE]

        response = {
            "file_name": Path(file_path).name,
            "topics": content.get("topics", []),
            "languages": valid_langs,
            "ocr_contents": content.get("ocr_contents", {})
        }
        return f"```json\n{json.dumps(response, indent=2, ensure_ascii=False)}\n```"

def create_interface():
    css = """
    .output-markdown {font-size: 14px; max-height: 500px; overflow-y: auto;}
    .status {color: #666; font-style: italic;}
    """
    
    with gr.Blocks(title="Mistral OCR App", css=css) as demo:
        gr.Markdown("# Mistral OCR App\nUpload images or PDFs for OCR processing")
        
        with gr.Row():
            api_key = gr.Textbox(label="Mistral API Key", type="password", placeholder="Enter your API key")
            set_key_btn = gr.Button("Set API Key", variant="primary")
        
        processor_state = gr.State()
        status = gr.Markdown("Please enter API key", elem_classes="status")

        def init_processor(key):
            try:
                processor = OCRProcessor(key)
                return processor, "✅ API key validated successfully"
            except Exception as e:
                return None, f"❌ Error: {str(e)}"

        set_key_btn.click(
            fn=init_processor,
            inputs=api_key,
            outputs=[processor_state, status]
        )

        with gr.Tab("Image OCR"):
            with gr.Row():
                image_input = gr.File(
                    label=f"Upload Image (max {MAX_FILE_SIZE/1024/1024}MB)",
                    file_types=SUPPORTED_IMAGE_TYPES
                )
                image_preview = gr.Image(label="Preview", height=300)
            image_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            process_image_btn = gr.Button("Process Image", variant="primary")

            def process_image(processor, image):
                if not processor or not image:
                    return "Please set API key and upload an image", None
                return processor.ocr_uploaded_image(image)

            process_image_btn.click(
                fn=process_image,
                inputs=[processor_state, image_input],
                outputs=[image_output, image_preview]
            )

        with gr.Tab("PDF OCR"):
            with gr.Row():
                pdf_input = gr.File(
                    label=f"Upload PDF (max {MAX_FILE_SIZE/1024/1024}MB, {MAX_PDF_PAGES} pages)",
                    file_types=SUPPORTED_PDF_TYPES
                )
                pdf_gallery = gr.Gallery(label="PDF Pages", height=300)
            pdf_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            process_pdf_btn = gr.Button("Process PDF", variant="primary")

            def process_pdf(processor, pdf):
                if not processor or not pdf:
                    return "Please set API key and upload a PDF", []
                return processor.ocr_uploaded_pdf(pdf)

            process_pdf_btn.click(
                fn=process_pdf,
                inputs=[processor_state, pdf_input],
                outputs=[pdf_output, pdf_gallery]
            )

        with gr.Tab("Structured OCR"):
            structured_input = gr.File(
                label=f"Upload Image for Structured OCR (max {MAX_FILE_SIZE/1024/1024}MB)",
                file_types=SUPPORTED_IMAGE_TYPES
            )
            structured_output = gr.Markdown(label="Structured Result", elem_classes="output-markdown")
            structured_preview = gr.Image(label="Preview", height=300)
            process_structured_btn = gr.Button("Process Structured OCR", variant="primary")

            def process_structured(processor, image):
                if not processor or not image:
                    return "Please set API key and upload an image", None
                return processor.structured_ocr(image)

            process_structured_btn.click(
                fn=process_structured,
                inputs=[processor_state, structured_input],
                outputs=[structured_output, structured_preview]
            )

    return demo

if __name__ == "__main__":
    os.environ['START_TIME'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== Application Startup at {os.environ['START_TIME']} =====")
    create_interface().launch(
        share=True,
        debug=True,
    )