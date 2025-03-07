import os
import base64
import gradio as gr
import json
import re  # Added to fix NameError
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from typing import Union, List, Tuple, Dict
import requests
import shutil
import time
import pymupdf as fitz
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import socket
from requests.exceptions import ConnectionError, Timeout
from pathlib import Path
from pydantic import BaseModel
import pycountry
from enum import Enum
from PIL import Image

# Constants
SUPPORTED_IMAGE_TYPES = [".jpg", ".png", ".jpeg"]
SUPPORTED_PDF_TYPES = [".pdf"]
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

# Language Enum for StructuredOCR
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

    def model_dump_json(self, **kwargs):
        data = self.model_dump(exclude_unset=True, by_alias=True, mode='json')
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, Language) for item in value):
                data[key] = [item.value for item in value]
        return json.dumps(data, indent=4)

class OCRProcessor:
    def __init__(self, api_key: str):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid API key must be provided")
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
        self._validate_client()

    def _validate_client(self) -> None:
        try:
            models = self.client.models.list()
            if not models:
                raise ValueError("No models available")
            logger.info("API key validated successfully")
        except Exception as e:
            raise ValueError(f"API key validation failed: {str(e)}")

    @staticmethod
    def _check_file_size(file_input: Union[str, bytes]) -> None:
        if isinstance(file_input, str) and os.path.exists(file_input):
            size = os.path.getsize(file_input)
        elif hasattr(file_input, 'read'):
            size = len(file_input.read())
            file_input.seek(0)
        else:
            size = len(file_input)
        if size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")

    @staticmethod
    def _save_uploaded_file(file_input: Union[str, bytes], filename: str) -> str:
        clean_filename = os.path.basename(filename).replace(os.sep, "_")
        file_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{clean_filename}")
        
        try:
            if isinstance(file_input, str) and file_input.startswith("http"):
                logger.info(f"Downloading from URL: {file_input}")
                response = requests.get(file_input, timeout=30)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            elif isinstance(file_input, str) and os.path.exists(file_input):
                logger.info(f"Copying local file: {file_input}")
                shutil.copy2(file_input, file_path)
            else:
                logger.info(f"Saving file object: {filename}")
                with open(file_path, 'wb') as f:
                    if hasattr(file_input, 'read'):
                        shutil.copyfileobj(file_input, f)
                    else:
                        f.write(file_input)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Failed to save file at {file_path}")
            logger.info(f"File saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            raise

    @staticmethod
    def _encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info(f"Encoded image {image_path} to base64 (length: {len(encoded)})")
                return encoded
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise ValueError(f"Failed to encode image: {str(e)}")

    @staticmethod
    def _pdf_to_images(pdf_path: str) -> List[Tuple[str, str]]:
        try:
            pdf_document = fitz.open(pdf_path)
            if pdf_document.page_count > MAX_PDF_PAGES:
                pdf_document.close()
                raise ValueError(f"PDF exceeds maximum page limit of {MAX_PDF_PAGES}")
            
            with ThreadPoolExecutor() as executor:
                image_data = list(executor.map(
                    lambda i: OCRProcessor._convert_page(pdf_path, i),
                    range(pdf_document.page_count)
                ))
            pdf_document.close()
            valid_image_data = [(path, encoded) for path, encoded in image_data if path and encoded]
            if not valid_image_data:
                raise ValueError("No valid pages converted from PDF")
            logger.info(f"Converted {len(valid_image_data)} pages to images")
            return valid_image_data
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    @staticmethod
    def _convert_page(pdf_path: str, page_num: int) -> Tuple[str, str]:
        try:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document[page_num]
            pix = page.get_pixmap(dpi=150)
            image_path = os.path.join(UPLOAD_FOLDER, f"page_{page_num + 1}_{int(time.time())}.png")
            pix.save(image_path)
            encoded = OCRProcessor._encode_image(image_path)
            pdf_document.close()
            return image_path, encoded
        except Exception as e:
            logger.error(f"Error converting page {page_num}: {str(e)}")
            return None, None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_ocr_api(self, encoded_image: str) -> OCRResponse:
        logger.info(f"Calling OCR API with API key: {self.api_key[:4]}...")  # Log partial key for debugging
        if not isinstance(encoded_image, str):
            raise TypeError(f"Expected encoded_image to be a string, got {type(encoded_image)}")
        base64_url = f"data:image/png;base64,{encoded_image}"
        try:
            response = self.client.ocr.process(
                document=ImageURLChunk(image_url=base64_url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            logger.info("OCR API call successful")
            try:
                if hasattr(response, 'model_dump_json'):
                    response_dict = json.loads(response.model_dump_json())
                else:
                    response_dict = {k: v for k, v in response.__dict__.items() if isinstance(v, (str, int, float, list, dict))}
                logger.info(f"Raw OCR response: {json.dumps(response_dict, default=str, indent=4)}")
            except Exception as log_err:
                logger.warning(f"Failed to log raw OCR response: {str(log_err)}")
            return response
        except Exception as e:
            logger.error(f"OCR API error: {str(e)}", exc_info=True)
            raise

    def _process_pdf_with_ocr(self, pdf_path: str) -> Tuple[str, List[str], List[Dict]]:
        try:
            logger.info(f"Processing PDF with API key: {self.api_key[:4]}...")
            uploaded_file = self.client.files.upload(
                file={"file_name": Path(pdf_path).stem, "content": Path(pdf_path).read_bytes()},
                purpose="ocr",
            )
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1).url

            ocr_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            markdown, base64_images = self._get_combined_markdown(ocr_response)
            json_results = self._convert_to_structured_json(markdown, pdf_path)
            image_paths = []
            if not any(page.images for page in ocr_response.pages):
                logger.warning("No images found in OCR response; using local images")
                image_data = self._pdf_to_images(pdf_path)
                image_paths = [path for path, _ in image_data]
            else:
                image_paths = [os.path.join(UPLOAD_FOLDER, f"ocr_page_{i}.png") for i in range(len(ocr_response.pages))]
                for i, base64_img in enumerate(base64_images):
                    if base64_img:
                        try:
                            img_data = base64.b64decode(base64_img.split(',')[1])
                            with open(image_paths[i], "wb") as f:
                                f.write(img_data)
                            if os.path.exists(image_paths[i]):
                                logger.info(f"Image {image_paths[i]} saved and exists")
                            else:
                                logger.error(f"Image {image_paths[i]} saved but does not exist")
                        except Exception as e:
                            logger.error(f"Error saving image {i}: {str(e)}")
                            image_paths[i] = None
                image_paths = [path for path in image_paths if path and os.path.exists(path)]
            logger.info(f"Final image paths: {image_paths}")
            return markdown, image_paths, json_results
        except Exception as e:
            return self._handle_error("PDF OCR processing", e), [], []

    def _get_combined_markdown(self, ocr_response: OCRResponse) -> Tuple[str, List[str]]:
        markdowns = []
        base64_images = []
        for i, page in enumerate(ocr_response.pages):
            image_data = {}
            for img in page.images:
                if img.image_base64:
                    # Use correct MIME type based on image format (assuming JPEG from logs)
                    base64_url = f"data:image/jpeg;base64,{img.image_base64}"
                    image_data[img.id] = base64_url
                    base64_images.append(base64_url)
                    logger.info(f"Base64 image {img.id} length: {len(img.image_base64)}")
                else:
                    base64_images.append(None)
            markdown = page.markdown or "No text detected"
            markdown = replace_images_in_markdown(markdown, image_data)
            logger.info(f"Page {i} markdown (first 200 chars): {markdown[:200]}...")
            markdowns.append(markdown)
        return "\n\n".join(markdowns), base64_images

    def _convert_to_structured_json(self, markdown: str, file_path: str) -> List[Dict]:
        try:
            text_only_markdown = re.sub(r'!\[.*?\]\(data:image/[^)]+\)', '', markdown)
            logger.info(f"Text-only markdown length: {len(text_only_markdown)}")
            logger.info(f"Text-only markdown content: {text_only_markdown[:200]}...")

            chat_response = self.client.chat.parse(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": f"Given OCR output from a PDF about African history and artifacts, convert to JSON with file_name, topics (e.g., African Artifacts, Tribal History), languages (e.g., English), and ocr_contents (title and list of items with descriptions and image refs).\n\nOCR Output:\n{text_only_markdown}"
                    },
                ],
                response_format=StructuredOCR,
                temperature=0
            )
            structured_result = chat_response.choices[0].message.parsed
            json_str = structured_result.model_dump_json()
            logger.info(f"Structured JSON: {json_str}")
            return [json.loads(json_str)]
        except Exception as e:
            logger.error(f"Error converting to structured JSON: {str(e)}", exc_info=True)
            return [{"error": str(e), "file_name": Path(file_path).stem}]

    def ocr_uploaded_pdf(self, pdf_file: Union[str, bytes]) -> Tuple[str, List[str], List[Dict]]:
        file_path = self._save_uploaded_file(pdf_file, getattr(pdf_file, 'name', f"pdf_{int(time.time())}.pdf"))
        return self._process_pdf_with_ocr(file_path)

    def ocr_pdf_url(self, pdf_url: str) -> Tuple[str, List[str], List[Dict]]:
        file_path = self._save_uploaded_file(pdf_url, pdf_url.split('/')[-1] or f"pdf_{int(time.time())}.pdf")
        return self._process_pdf_with_ocr(file_path)

    def ocr_uploaded_image(self, image_file: Union[str, bytes]) -> Tuple[str, str, Dict]:
        file_path = self._save_uploaded_file(image_file, getattr(image_file, 'name', f"image_{int(time.time())}.jpg"))
        encoded_image = self._encode_image(file_path)
        base64_url = f"data:image/png;base64,{encoded_image}"
        response = self._call_ocr_api(encoded_image)
        markdown, base64_images = self._get_combined_markdown(response)
        json_result = self._convert_to_structured_json(markdown, file_path)[0]
        return markdown, file_path, json_result

    @staticmethod
    def _handle_error(context: str, error: Exception) -> str:
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        return f"**Error in {context}:** {str(error)}"

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def create_interface():
    css = """
    .output-markdown {font-size: 14px; max-height: 500px; overflow-y: auto;}
    .status {color: #666; font-style: italic;}
    """
    
    with gr.Blocks(title="Mistral OCR App", css=css) as demo:
        gr.Markdown("# Mistral OCR App\nUpload images or PDFs, or provide a PDF URL for OCR processing")
        
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
            image_json_output = gr.JSON(label="Structured JSON Output")
            process_image_btn = gr.Button("Process Image", variant="primary")

            def process_image(processor, image):
                if not processor or not image:
                    return "Please set API key and upload an image", None, {}
                markdown, image_path, json_data = processor.ocr_uploaded_image(image)
                return markdown, image_path, json_data

            process_image_btn.click(
                fn=process_image,
                inputs=[processor_state, image_input],
                outputs=[image_output, image_preview, image_json_output]
            )

        with gr.Tab("PDF OCR"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label=f"Upload PDF (max {MAX_FILE_SIZE/1024/1024}MB, {MAX_PDF_PAGES} pages)",
                        file_types=SUPPORTED_PDF_TYPES
                    )
                    pdf_url_input = gr.Textbox(
                        label="Or Enter PDF URL",
                        placeholder="e.g., https://arxiv.org/pdf/2201.04234.pdf"
                    )
                pdf_gallery = gr.Gallery(label="PDF Pages", height=300)
            pdf_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            pdf_json_output = gr.JSON(label="Structured JSON Output")
            process_pdf_btn = gr.Button("Process PDF", variant="primary")

            def process_pdf(processor, pdf_file, pdf_url):
                if not processor:
                    return "Please set API key first", [], {}
                logger.info(f"Received inputs - PDF file: {pdf_file}, PDF URL: {pdf_url}")
                if pdf_file is not None and hasattr(pdf_file, 'name'):
                    logger.info(f"Processing as uploaded PDF: {pdf_file.name}")
                    markdown, image_paths, json_data = processor.ocr_uploaded_pdf(pdf_file)
                elif pdf_url and pdf_url.strip():
                    logger.info(f"Processing as PDF URL: {pdf_url}")
                    markdown, image_paths, json_data = processor.ocr_pdf_url(pdf_url)
                else:
                    return "Please upload a PDF or provide a valid URL", [], {}
                return markdown, image_paths, json_data

            process_pdf_btn.click(
                fn=process_pdf,
                inputs=[processor_state, pdf_input, pdf_url_input],
                outputs=[pdf_output, pdf_gallery, pdf_json_output]
            )

    return demo

if __name__ == "__main__":
    os.environ['START_TIME'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== Application Startup at {os.environ['START_TIME']} =====")
    create_interface().launch(
        share=True,
        debug=True,
    )