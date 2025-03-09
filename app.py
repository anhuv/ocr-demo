import os
import base64
import gradio as gr
import requests
import shutil
import time
import pymupdf as fitz
import logging
from mistralai import Mistral, ImageURLChunk
from mistralai.models import OCRResponse
from typing import Union, List, Tuple, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import tempfile

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
                response = requests.get(file_input, timeout=30)
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
    def _encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
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
            return [data for data in image_data if data]
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
            return response
        except Exception as e:
            logger.error(f"OCR API call failed: {str(e)}")
            raise

    def process_file(self, file: gr.File) -> Tuple[str, str, List[str]]:
        """Process uploaded file (image or PDF)."""
        if not file:
            return "## No file provided", "", []
        
        file_name = file.name
        self._check_file_size(file)
        file_path = self._save_uploaded_file(file, file_name)
        
        if file_name.lower().endswith(tuple(SUPPORTED_IMAGE_TYPES)):
            encoded_image = self._encode_image(file_path)
            response = self._call_ocr_api(encoded_image)
            markdown = self._combine_markdown(response)
            return markdown, file_path, [file_path]
        
        elif file_name.lower().endswith('.pdf'):
            image_data = self._pdf_to_images(file_path)
            if not image_data:
                return "## No pages converted from PDF", file_path, []
            
            ocr_results = []
            image_paths = [path for path, _ in image_data]
            for _, encoded in image_data:
                response = self._call_ocr_api(encoded)
                markdown = self._combine_markdown(response)
                ocr_results.append(markdown)
            return "\n\n".join(ocr_results), file_path, image_paths
        
        return "## Unsupported file type", file_path, []

    def process_url(self, url: str) -> Tuple[str, str, List[str]]:
        """Process URL (image or PDF)."""
        if not url:
            return "## No URL provided", "", []
        
        file_name = url.split('/')[-1] or f"file_{int(time.time())}"
        file_path = self._save_uploaded_file(url, file_name)
        
        if file_name.lower().endswith(tuple(SUPPORTED_IMAGE_TYPES)):
            encoded_image = self._encode_image(file_path)
            response = self._call_ocr_api(encoded_image)
            markdown = self._combine_markdown(response)
            return markdown, url, [file_path]
        
        elif file_name.lower().endswith('.pdf'):
            image_data = self._pdf_to_images(file_path)
            if not image_data:
                return "## No pages converted from PDF", url, []
            
            ocr_results = []
            image_paths = [path for path, _ in image_data]
            for _, encoded in image_data:
                response = self._call_ocr_api(encoded)
                markdown = self._combine_markdown(response)
                ocr_results.append(markdown)
            return "\n\n".join(ocr_results), url, image_paths
        
        return "## Unsupported URL content type", url, []

    @staticmethod
    def _combine_markdown(response: OCRResponse) -> str:
        """Combine markdown from OCR response."""
        markdown_parts = []
        for page in response.pages:
            if not page.markdown.strip():
                continue
            markdown = page.markdown
            if hasattr(page, 'images') and page.images:
                for img in page.images:
                    if img.image_base64:
                        markdown = markdown.replace(
                            f"![{img.id}]({img.id})",
                            f"![{img.id}](data:image/png;base64,{img.image_base64})"
                        )
            markdown_parts.append(markdown)
        return "\n\n".join(markdown_parts) or "## No text detected"

def create_interface():
    css = """
    .output-markdown {font-size: 14px; max-height: 500px; overflow-y: auto;}
    .status {color: #666; font-style: italic;}
    .preview {max-height: 300px;}
    """

    with gr.Blocks(title="Mistral OCR Demo", css=css) as demo:
        gr.Markdown("# Mistral OCR Demo")
        gr.Markdown(f"""
            Process PDFs and images (max {MAX_FILE_SIZE/1024/1024}MB, {MAX_PDF_PAGES} pages for PDFs) via upload or URL.
            View previews and OCR results with embedded images.
            Learn more at [Mistral OCR](https://mistral.ai/news/mistral-ocr).
        """)

        # API Key Setup
        with gr.Row():
            api_key_input = gr.Textbox(label="Mistral API Key", type="password", placeholder="Enter your API key")
            set_key_btn = gr.Button("Set API Key", variant="primary")
        processor_state = gr.State()
        status = gr.Markdown("Please enter API key", elem_classes="status")

        def init_processor(key):
            try:
                processor = OCRProcessor(key)
                return processor, "✅ API key validated"
            except Exception as e:
                return None, f"❌ Error: {str(e)}"
        
        set_key_btn.click(fn=init_processor, inputs=api_key_input, outputs=[processor_state, status])

        # File Upload Tab
        with gr.Tab("Upload File"):
            with gr.Row():
                file_input = gr.File(label="Upload PDF/Image", file_types=SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF_TYPES)
                file_preview = gr.Gallery(label="Preview", elem_classes="preview")
            file_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            file_raw_output = gr.Textbox(label="Raw File Path")
            file_button = gr.Button("Process", variant="primary")

            def update_file_preview(file):
                return [file.name] if file else []

            file_input.change(fn=update_file_preview, inputs=file_input, outputs=file_preview)
            file_button.click(
                fn=lambda p, f: p.process_file(f) if p else ("## Set API key first", "", []),
                inputs=[processor_state, file_input],
                outputs=[file_output, file_raw_output, file_preview]
            )

        # URL Tab
        with gr.Tab("URL Input"):
            with gr.Row():
                url_input = gr.Textbox(label="URL to PDF/Image")
                url_preview = gr.Gallery(label="Preview", elem_classes="preview")
            url_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            url_raw_output = gr.Textbox(label="Raw URL")
            url_button = gr.Button("Process", variant="primary")

            def update_url_preview(url):
                if not url:
                    return []
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
                    response = requests.get(url, timeout=10)
                    temp_file.write(response.content)
                    temp_file.close()
                    return [temp_file.name]
                except Exception as e:
                    logger.error(f"URL preview error: {str(e)}")
                    return []

            url_input.change(fn=update_url_preview, inputs=url_input, outputs=url_preview)
            url_button.click(
                fn=lambda p, u: p.process_url(u) if p else ("## Set API key first", "", []),
                inputs=[processor_state, url_input],
                outputs=[url_output, url_raw_output, url_preview]
            )

        # Examples
        gr.Examples(
            examples=[],
            inputs=[file_input, url_input]
        )

    return demo

if __name__ == "__main__":
    os.environ['START_TIME'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== Application Startup at {os.environ['START_TIME']} =====")
    create_interface().launch(share=True, max_threads=1)