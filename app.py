import os
import base64
import gradio as gr
from mistralai import Mistral, ImageURLChunk
from mistralai.models import OCRResponse
from typing import Union, List, Tuple
import requests
import shutil
import time
import pymupdf as fitz
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import socket
from requests.exceptions import ConnectionError, Timeout

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

class OCRProcessor:
    def __init__(self, api_key: str):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid API key must be provided")
        self.client = Mistral(api_key=api_key)
        self._validate_client()

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
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise ValueError("Failed to encode image")

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
            valid_data = [data for data in image_data if data and data[0] and os.path.exists(data[0])]
            if not valid_data:
                logger.warning("No valid images generated from PDF")
            return valid_data
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []

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
        base64_url = f"data:image/png;base64,{encoded_image}"
        try:
            logger.info("Calling OCR API")
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document=ImageURLChunk(image_url=base64_url),
                include_image_base64=True
            )
            logger.info(f"OCR API call successful. Pages: {len(response.pages)}")
            for page in response.pages:
                logger.debug(f"Page markdown: {page.markdown}")
            return response
        except (ConnectionError, Timeout, socket.error) as e:
            logger.error(f"Network error during OCR API call: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"OCR API error: {str(e)}")
            raise

    def ocr_uploaded_pdf(self, pdf_file: Union[str, bytes]) -> Tuple[str, List[str]]:
        file_name = getattr(pdf_file, 'name', f"pdf_{int(time.time())}.pdf")
        logger.info(f"Processing uploaded PDF: {file_name}")
        try:
            self._check_file_size(pdf_file)
            pdf_path = self._save_uploaded_file(pdf_file, file_name)
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Saved PDF not found at: {pdf_path}")
            
            image_data = self._pdf_to_images(pdf_path)
            if not image_data:
                return "No valid pages converted from PDF", []
            
            ocr_results = []
            image_paths = [path for path, _ in image_data]
            for i, (_, encoded) in enumerate(image_data):
                response = self._call_ocr_api(encoded)
                markdown_with_images = self._get_combined_markdown_with_images(response, image_paths, i)
                ocr_results.append(markdown_with_images)
            
            return "\n\n".join(ocr_results) or "No text detected in PDF", image_paths
        except Exception as e:
            return self._handle_error("uploaded PDF processing", e), []

    def ocr_pdf_url(self, pdf_url: str) -> Tuple[str, List[str]]:
        logger.info(f"Processing PDF URL: {pdf_url}")
        try:
            file_name = pdf_url.split('/')[-1] or f"pdf_{int(time.time())}.pdf"
            pdf_path = self._save_uploaded_file(pdf_url, file_name)
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Saved PDF not found at: {pdf_path}")
            
            image_data = self._pdf_to_images(pdf_path)
            if not image_data:
                return "No valid pages converted from PDF", []
            
            ocr_results = []
            image_paths = [path for path, _ in image_data]
            for i, (_, encoded) in enumerate(image_data):
                response = self._call_ocr_api(encoded)
                markdown_with_images = self._get_combined_markdown_with_images(response, image_paths, i)
                ocr_results.append(markdown_with_images)
            
            return "\n\n".join(ocr_results) or "No text detected in PDF", image_paths
        except Exception as e:
            return self._handle_error("PDF URL processing", e), []

    def ocr_uploaded_image(self, image_file: Union[str, bytes]) -> Tuple[str, gr.components.Image.update]:
        file_name = getattr(image_file, 'name', f"image_{int(time.time())}.jpg")
        logger.info(f"Processing uploaded image: {file_name}")
        try:
            self._check_file_size(image_file)
            image_path = self._save_uploaded_file(image_file, file_name)
            encoded_image = self._encode_image(image_path)
            response = self._call_ocr_api(encoded_image)
            markdown_with_images = self._get_combined_markdown_with_images(response)
            preview_update = gr.Image.update(value=image_path) if image_path else gr.Image.update()
            return markdown_with_images or "No text detected in image", preview_update
        except Exception as e:
            return self._handle_error("image processing", e), gr.Image.update()

    @staticmethod
    def _get_combined_markdown_with_images(response: OCRResponse, image_paths: List[str] = None, page_index: int = None) -> str:
        markdown_parts = []
        logger.info(f"Processing response with {len(response.pages)} pages")
        for i, page in enumerate(response.pages):
            if page.markdown and page.markdown.strip():
                markdown = page.markdown.strip()
                logger.info(f"Page {i} markdown: {markdown[:100]}...")  # Log first 100 chars
                if hasattr(page, 'images') and page.images:
                    logger.info(f"Found {len(page.images)} images in page {i}")
                    for img in page.images:
                        if img.image_base64:
                            logger.info(f"Replacing image {img.id} with base64")
                            markdown = markdown.replace(
                                f"![{img.id}]({img.id})",
                                f"![{img.id}](data:image/png;base64,{img.image_base64})"
                            )
                        else:
                            logger.warning(f"No base64 data for image {img.id}")
                            if image_paths and page_index is not None and page_index < len(image_paths):
                                local_encoded = OCRProcessor._encode_image(image_paths[page_index])
                                markdown = markdown.replace(
                                    f"![{img.id}]({img.id})",
                                    f"![{img.id}](data:image/png;base64,{local_encoded})"
                                )
                else:
                    logger.warning(f"No images found in page {i}")
                    if image_paths and page_index is not None and page_index < len(image_paths):
                        local_encoded = OCRProcessor._encode_image(image_paths[page_index])
                        placeholder = f"img-{i}.jpeg"
                        if placeholder in markdown:
                            markdown = markdown.replace(
                                placeholder,
                                f"![Page {i} Image](data:image/png;base64,{local_encoded})"
                            )
                        else:
                            markdown += f"\n\n![Page {i} Image](data:image/png;base64,{local_encoded})"
                markdown_parts.append(markdown)
            else:
                logger.warning(f"No markdown content in page {i}")
        return "\n\n".join(markdown_parts) or "No text or images detected"

    @staticmethod
    def _handle_error(context: str, error: Exception) -> str:
        logger.error(f"Error in {context}: {str(error)}")
        return f"**Error in {context}:** {str(error)}"

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
            process_image_btn = gr.Button("Process Image", variant="primary")

            def process_image(processor, image):
                if not processor or not image:
                    return "Please set API key and upload an image", gr.Image.update()
                result, preview_update = processor.ocr_uploaded_image(image)
                return result, preview_update

            process_image_btn.click(
                fn=process_image,
                inputs=[processor_state, image_input],
                outputs=[image_output, image_preview]
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
            process_pdf_btn = gr.Button("Process PDF", variant="primary")

            def process_pdf(processor, pdf_file, pdf_url):
                if not processor:
                    return "Please set API key first", gr.Gallery.update()
                logger.info(f"Received inputs - PDF file: {pdf_file}, PDF URL: {pdf_url}")
                if pdf_file is not None and hasattr(pdf_file, 'name'):
                    logger.info(f"Processing as uploaded PDF: {pdf_file.name}")
                    result, image_paths = processor.ocr_uploaded_pdf(pdf_file)
                    gallery = gr.Gallery.update(value=[(p, os.path.basename(p)) for p in image_paths]) if image_paths else gr.Gallery.update()
                    return result, gallery
                elif pdf_url and pdf_url.strip():
                    logger.info(f"Processing as PDF URL: {pdf_url}")
                    result, image_paths = processor.ocr_pdf_url(pdf_url)
                    gallery = gr.Gallery.update(value=[(p, os.path.basename(p)) for p in image_paths]) if image_paths else gr.Gallery.update()
                    return result, gallery
                return "Please upload a PDF or provide a valid URL", gr.Gallery.update()

            process_pdf_btn.click(
                fn=process_pdf,
                inputs=[processor_state, pdf_input, pdf_url_input],
                outputs=[pdf_output, pdf_gallery]
            )

    return demo

if __name__ == "__main__":
    os.environ['START_TIME'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== Application Startup at {os.environ['START_TIME']} =====")
    create_interface().launch(
        share=True,
    )