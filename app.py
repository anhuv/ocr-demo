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
        return self.client.ocr.process(
            model="mistral-ocr-latest",
            document=ImageURLChunk(image_url=base64_url),
            include_image_base64=True
        )

    def ocr_uploaded_pdf(self, pdf_file: Union[str, bytes]) -> Tuple[str, List[str]]:
        file_name = getattr(pdf_file, 'name', f"pdf_{int(time.time())}.pdf")
        logger.info(f"Processing uploaded PDF: {file_name}")
        try:
            self._check_file_size(pdf_file)
            pdf_path = self._save_uploaded_file(pdf_file, file_name)
            logger.info(f"Saved PDF to: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Saved PDF not found at: {pdf_path}")
            
            image_data = self._pdf_to_images(pdf_path)
            if not image_data:
                raise ValueError("No pages converted from PDF")
            
            # Process each page with OCR
            ocr_results = []
            for _, encoded in image_data:
                response = self._call_ocr_api(encoded)
                markdown = self._get_combined_markdown(response)
                ocr_results.append(markdown)
            
            image_paths = [path for path, _ in image_data]
            return "\n\n".join(ocr_results), image_paths
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
                raise ValueError("No pages converted from PDF")
            
            ocr_results = []
            for _, encoded in image_data:
                response = self._call_ocr_api(encoded)
                markdown = self._get_combined_markdown(response)
                ocr_results.append(markdown)
            
            image_paths = [path for path, _ in image_data]
            return "\n\n".join(ocr_results), image_paths
        except Exception as e:
            return self._handle_error("PDF URL processing", e), []

    def ocr_uploaded_image(self, image_file: Union[str, bytes]) -> Tuple[str, str]:
        file_name = getattr(image_file, 'name', f"image_{int(time.time())}.jpg")
        logger.info(f"Processing uploaded image: {file_name}")
        try:
            self._check_file_size(image_file)
            image_path = self._save_uploaded_file(image_file, file_name)
            encoded_image = self._encode_image(image_path)
            response = self._call_ocr_api(encoded_image)
            return self._get_combined_markdown(response), image_path
        except Exception as e:
            return self._handle_error("image processing", e), None

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
                    return "Please set API key and upload an image", None
                return processor.ocr_uploaded_image(image)

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
                    return "Please set API key first", []
                if pdf_file:
                    return processor.ocr_uploaded_pdf(pdf_file)
                elif pdf_url:
                    return processor.ocr_pdf_url(pdf_url)
                return "Please upload a PDF or provide a URL", []

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
        debug=True,
    )