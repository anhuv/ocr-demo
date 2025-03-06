import os
import base64
import gradio as gr
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from pathlib import Path
import pycountry
import json
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import tempfile
from typing import Union, Dict, List
from contextlib import contextmanager
import requests
import shutil

# Constants
DEFAULT_LANGUAGE = "English"
SUPPORTED_IMAGE_TYPES = [".jpg", ".png"]
SUPPORTED_PDF_TYPES = [".pdf"]
TEMP_FILE_EXPIRY = 7200  # 2 hours in seconds
UPLOAD_FOLDER = "uploads"  # Local storage folder

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key must be provided")
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        try:
            models = self.client.models.list()  # Validate API key
            if not models:
                raise ValueError("No models available")
        except Exception as e:
            raise ValueError(f"Invalid API key: {str(e)}")

    @staticmethod
    def _encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None

    @staticmethod
    def _save_uploaded_file(file_input: Union[str, bytes], filename: str) -> str:
        """Save uploaded file to local storage and return path"""
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if isinstance(file_input, str):
                if file_input.startswith("http"):
                    response = requests.get(file_input)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    # Copy file to new location if source and destination are different
                    if os.path.abspath(file_input) != os.path.abspath(file_path):
                        shutil.copy2(file_input, file_path)
                    else:
                        return file_input  # Return original path if same file
            else:
                with open(file_path, 'wb') as f:
                    if hasattr(file_input, 'read'):
                        shutil.copyfileobj(file_input, f)
                    else:
                        f.write(file_input)
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return None

    @staticmethod
    def _pdf_to_images(pdf_path: str) -> List[str]:
        """Convert PDF pages to images and return their paths"""
        image_paths = []
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                image_path = os.path.join(UPLOAD_FOLDER, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
            pdf_document.close()
            return image_paths
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []

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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call_ocr_api(self, document: Union[DocumentURLChunk, ImageURLChunk]) -> OCRResponse:
        try:
            return self.client.ocr.process(model="mistral-ocr-latest", document=document, include_image_base64=True)
        except Exception as e:
            logger.error(f"OCR API call failed: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call_chat_complete(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        try:
            return self.client.chat.complete(model=model, messages=messages, **kwargs)
        except Exception as e:
            logger.error(f"Chat complete API call failed: {str(e)}")
            raise

    def _get_file_content(self, file_input: Union[str, bytes]) -> bytes:
        if isinstance(file_input, str):
            if file_input.startswith("http"):
                response = requests.get(file_input)
                response.raise_for_status()
                return response.content
            else:
                with open(file_input, "rb") as f:
                    return f.read()
        return file_input.read() if hasattr(file_input, 'read') else file_input

    def ocr_pdf_url(self, pdf_url: str) -> tuple[str, List[str]]:
        logger.info(f"Processing PDF URL: {pdf_url}")
        try:
            # Download and save PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            filename = pdf_url.split('/')[-1]
            pdf_path = self._save_uploaded_file(response.content, filename)
            if not pdf_path:
                return self._handle_error("PDF saving", Exception("Failed to save PDF")), []
            
            # Convert PDF to images for visualization
            image_paths = self._pdf_to_images(pdf_path)
            
            # Process with OCR
            response = self._call_ocr_api(DocumentURLChunk(document_url=pdf_url))
            return self._get_combined_markdown(response), image_paths
        except Exception as e:
            return self._handle_error("PDF URL processing", e), []

    def ocr_uploaded_pdf(self, pdf_file: Union[str, bytes]) -> tuple[str, List[str]]:
        file_name = getattr(pdf_file, 'name', 'unknown')
        logger.info(f"Processing uploaded PDF: {file_name}")
        try:
            # Save uploaded PDF
            pdf_path = self._save_uploaded_file(pdf_file, file_name)
            if not pdf_path:
                return self._handle_error("PDF saving", Exception("Failed to save PDF")), []
            
            # Convert PDF to images for visualization
            image_paths = self._pdf_to_images(pdf_path)
            
            # Process with OCR
            uploaded_file = self.client.files.upload(
                file={"file_name": pdf_path, "content": open(pdf_path, "rb")},
                purpose="ocr"
            )
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=TEMP_FILE_EXPIRY)
            response = self._call_ocr_api(DocumentURLChunk(document_url=signed_url.url))
            return self._get_combined_markdown(response), image_paths
        except Exception as e:
            return self._handle_error("uploaded PDF processing", e), []

    def ocr_image_url(self, image_url: str) -> tuple[str, str]:
        logger.info(f"Processing image URL: {image_url}")
        try:
            # Download and save image
            response = requests.get(image_url)
            response.raise_for_status()
            filename = image_url.split('/')[-1]
            image_path = self._save_uploaded_file(response.content, filename)
            if not image_path:
                return self._handle_error("image saving", Exception("Failed to save image")), None
            
            # Process with OCR
            response = self._call_ocr_api(ImageURLChunk(image_url=image_url))
            return self._get_combined_markdown(response), image_path
        except Exception as e:
            return self._handle_error("image URL processing", e), None

    def ocr_uploaded_image(self, image_file: Union[str, bytes]) -> tuple[str, str]:
        file_name = getattr(image_file, 'name', 'unknown')
        logger.info(f"Processing uploaded image: {file_name}")
        try:
            # Save uploaded image
            image_path = self._save_uploaded_file(image_file, file_name)
            if not image_path:
                return self._handle_error("image saving", Exception("Failed to save image")), None
            
            # Process with OCR
            encoded_image = self._encode_image(image_path)
            if encoded_image is None:
                return self._handle_error("image encoding", Exception("Failed to encode image")), None
            base64_url = f"data:image/jpeg;base64,{encoded_image}"
            response = self._call_ocr_api(ImageURLChunk(image_url=base64_url))
            return self._get_combined_markdown(response), image_path
        except Exception as e:
            return self._handle_error("uploaded image processing", e), None

    def document_understanding(self, doc_url: str, question: str) -> str:
        logger.info(f"Document understanding - URL: {doc_url}, Question: {question}")
        try:
            messages = [{"role": "user", "content": [
                TextChunk(text=question),
                DocumentURLChunk(document_url=doc_url)
            ]}]
            response = self._call_chat_complete(model="mistral-small-latest", messages=messages)
            return response.choices[0].message.content if response.choices else "No response received"
        except Exception as e:
            return self._handle_error("document understanding", e)

    def structured_ocr(self, image_file: Union[str, bytes]) -> tuple[str, str]:
        file_name = getattr(image_file, 'name', 'unknown')
        logger.info(f"Processing structured OCR for: {file_name}")
        try:
            # Save uploaded image
            image_path = self._save_uploaded_file(image_file, file_name)
            if not image_path:
                return self._handle_error("image saving", Exception("Failed to save image")), None
            
            encoded_image = self._encode_image(image_path)
            if encoded_image is None:
                return self._handle_error("image encoding", Exception("Failed to encode image")), None
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
                            "Convert this into a sensible structured json response with file_name, topics, languages, and ocr_contents fields"
                        ))
                    ]
                }],
                response_format={"type": "json_object"},
                temperature=0
            )

            response_content = chat_response.choices[0].message.content
            content = json.loads(response_content)
            return self._format_structured_response(image_path, content), image_path
        except Exception as e:
            return self._handle_error("structured OCR", e), None

    def _get_combined_markdown(self, response: OCRResponse) -> str:
        markdowns = []
        for page in response.pages:
            image_data = {}
            for img in page.images:
                image_data[img.id] = img.image_base64
            markdown = page.markdown
            for img_name, base64_str in image_data.items():
                markdown = markdown.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
            markdowns.append(markdown)
        return "\n\n".join(markdowns)

    @staticmethod
    def _handle_error(context: str, error: Exception) -> str:
        logger.error(f"Error in {context}: {str(error)}")
        return f"**Error:** {str(error)}"

    @staticmethod
    def _format_structured_response(file_path: str, content: Dict) -> str:
        languages = {lang.alpha_2: lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')}
        # Handle languages as a list instead of using .get()
        content_languages = content["languages"] if "languages" in content else [DEFAULT_LANGUAGE]
        valid_langs = [l for l in content_languages if l in languages.values()]

        response = {
            "file_name": Path(file_path).name,
            "topics": content["topics"] if "topics" in content else [],
            "languages": valid_langs or [DEFAULT_LANGUAGE],
            "ocr_contents": content["ocr_contents"] if "ocr_contents" in content else {}
        }
        return f"```json\n{json.dumps(response, indent=4)}\n```"

def create_interface():
    with gr.Blocks(title="Mistral OCR App") as demo:
        gr.Markdown("# Mistral OCR App")
        
        api_key = gr.Textbox(label="API Key", type="password")
        processor_state = gr.State()
        status = gr.Markdown()

        def init_processor(key):
            try:
                processor = OCRProcessor(key)
                return processor, "API key validated!"
            except Exception as e:
                return None, f"Error: {str(e)}"

        gr.Button("Set API Key").click(
            fn=init_processor,
            inputs=api_key,
            outputs=[processor_state, status]
        )

        with gr.Tab("Image OCR"):
            image_input = gr.File(label="Upload Image", file_types=SUPPORTED_IMAGE_TYPES)
            image_preview = gr.Image(label="Image Preview")
            image_output = gr.Markdown()

            def process_image(processor, image):
                if not processor:
                    return "Please set API key first", None
                ocr_result, image_path = processor.ocr_uploaded_image(image)
                return ocr_result, image_path

            gr.Button("Process Image").click(
                fn=process_image,
                inputs=[processor_state, image_input],
                outputs=[image_output, image_preview]
            )

        with gr.Tab("PDF OCR"):
            pdf_input = gr.File(label="Upload PDF", file_types=SUPPORTED_PDF_TYPES)
            pdf_gallery = gr.Gallery(label="PDF Pages")
            pdf_output = gr.Markdown()

            def process_pdf(processor, pdf):
                if not processor:
                    return "Please set API key first", None
                ocr_result, image_paths = processor.ocr_uploaded_pdf(pdf)
                return ocr_result, image_paths

            gr.Button("Process PDF").click(
                fn=process_pdf,
                inputs=[processor_state, pdf_input],
                outputs=[pdf_output, pdf_gallery]
            )

    return demo