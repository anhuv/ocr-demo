import os
import base64
import gradio as gr
import requests
import shutil
import time
import pymupdf as fitz
import logging
import mimetypes
from mistralai import Mistral
from mistralai.models import OCRResponse
from typing import Union, List, Tuple, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
import tempfile

# Constants
SUPPORTED_IMAGE_TYPES = [".jpg", ".png", ".jpeg", ".avif"]
SUPPORTED_DOCUMENT_TYPES = [".pdf"]
UPLOAD_FOLDER = "./uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PDF_PAGES = 50  # Not used anymore, kept for reference

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
        self.file_ids_to_delete = []
        self._validate_client()

    def _validate_client(self) -> None:
        try:
            models = self.client.models.list()
            if not models:
                raise ValueError("No models available")
        except Exception as e:
            raise ValueError(f"API key validation failed: {str(e)}")

    @staticmethod
    def _check_file_size(file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        size = os.path.getsize(file_path)
        if size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit")

    def _upload_file_for_ocr(self, file_path: str) -> str:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                uploaded_file = self.client.files.upload(
                    file={"file_name": filename, "content": f},
                    purpose="ocr"
                )
            self.file_ids_to_delete.append(uploaded_file.id)
            signed_url = self.client.files.get_signed_url(uploaded_file.id)
            return signed_url.url
        except Exception as e:
            logger.error(f"Failed to upload file {filename}: {str(e)}")
            raise ValueError(f"Failed to upload file: {str(e)}")

    @staticmethod
    def _convert_first_page(pdf_path: str) -> Optional[str]:
        try:
            pdf_document = fitz.open(pdf_path)
            if pdf_document.page_count == 0:
                pdf_document.close()
                return None
            page = pdf_document[0]
            pix = page.get_pixmap(dpi=100)
            img_path = os.path.join(UPLOAD_FOLDER, f"preview_{int(time.time())}.png")
            pix.save(img_path)
            pdf_document.close()
            return img_path
        except Exception as e:
            logger.error(f"Error converting first page of {pdf_path}: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_ocr_api(self, document: Dict) -> OCRResponse:
        try:
            logger.info("Calling OCR API")
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document=document,
                include_image_base64=True
            )
            return response
        except Exception as e:
            logger.error(f"OCR API call failed: {str(e)}")
            raise

    def process_file(self, file: gr.File) -> Tuple[str, str]:
        """Process uploaded file (image or PDF)."""
        if not file:
            return "## No file provided", ""
        file_path = file.name
        self._check_file_size(file_path)
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        try:
            if ext in SUPPORTED_IMAGE_TYPES:
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    mime_type = "image/png"
                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                data_url = f"data:{mime_type};base64,{base64_encoded}"
                document = {"type": "image_url", "image_url": data_url}
                response = self._call_ocr_api(document)
                markdown = self._combine_markdown(response)
                return markdown, file_path
            elif ext in SUPPORTED_DOCUMENT_TYPES:
                signed_url = self._upload_file_for_ocr(file_path)
                document = {"type": "document_url", "document_url": signed_url}
                response = self._call_ocr_api(document)
                markdown = self._combine_markdown(response)
                return markdown, file_path
            else:
                return f"## Unsupported file type. Supported: {', '.join(SUPPORTED_IMAGE_TYPES + SUPPORTED_DOCUMENT_TYPES)}", file_path
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            return f"## Error processing file: {str(e)}", file_path

    def process_url(self, url: str) -> Tuple[str, str]:
        """Process URL (image or PDF)."""
        if not url:
            return "## No URL provided", ""
        parsed_url = url.split('/')[-1] if '/' in url else url
        ext = os.path.splitext(parsed_url)[1].lower()
        try:
            if ext in SUPPORTED_IMAGE_TYPES:
                document = {"type": "image_url", "image_url": url}
                response = self._call_ocr_api(document)
                markdown = self._combine_markdown(response)
                return markdown, url
            elif ext in SUPPORTED_DOCUMENT_TYPES:
                document = {"type": "document_url", "document_url": url}
                response = self._call_ocr_api(document)
                markdown = self._combine_markdown(response)
                return markdown, url
            else:
                return f"## Unsupported URL type. Supported: {', '.join(SUPPORTED_IMAGE_TYPES + SUPPORTED_DOCUMENT_TYPES)}", url
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return f"## Error processing URL: {str(e)}", url

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

def update_file_preview(file):
    if not file:
        return gr.update(value=[])
    ext = os.path.splitext(os.path.basename(file.name))[1].lower()
    if ext in SUPPORTED_IMAGE_TYPES:
        return gr.update(value=[file.name])
    elif ext in SUPPORTED_DOCUMENT_TYPES:
        first_page = OCRProcessor._convert_first_page(file.name)
        return gr.update(value=[first_page] if first_page else [])
    else:
        return gr.update(value=[])

def update_url_preview(url):
    if not url:
        return gr.update(value=[])
    parsed_url = url.split('/')[-1] if '/' in url else url
    ext = os.path.splitext(parsed_url)[1].lower()
    if ext in SUPPORTED_IMAGE_TYPES:
        return gr.update(value=[url])
    elif ext == '.pdf':  # Only preview PDFs
        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                shutil.copyfileobj(response.raw, temp_pdf)
                temp_pdf_path = temp_pdf.name
            first_page = OCRProcessor._convert_first_page(temp_pdf_path)
            os.unlink(temp_pdf_path)
            return gr.update(value=[first_page] if first_page else [])
        except Exception as e:
            logger.error(f"URL preview error: {str(e)}")
            return gr.update(value=[])
    else:
        return gr.update(value=[])

def create_interface():
    css = """
    .output-markdown {font-size: 14px; max-height: 500px; overflow-y: auto;}
    .status {color: #666; font-style: italic;}
    .preview {max-height: 300px;}
    """
    with gr.Blocks(title="Mistral OCR Demo", css=css) as demo:
        gr.Markdown("# Mistral OCR Demo")
        gr.Markdown(f"""Process PDFs and images (max {MAX_FILE_SIZE/1024/1024}MB) via upload or URL. 
        Supported: Images ({', '.join(SUPPORTED_IMAGE_TYPES)}), Documents ({', '.join(SUPPORTED_DOCUMENT_TYPES)}). 
        View previews and OCR results with embedded images. 
        Learn more at [Mistral OCR](https://docs.mistral.ai/capabilities/document_ai/basic_ocr).""")

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
                file_input = gr.File(label="Upload Image/PDF", file_types=SUPPORTED_IMAGE_TYPES + SUPPORTED_DOCUMENT_TYPES)
            file_preview = gr.Gallery(label="Preview", elem_classes="preview")
            file_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            file_raw_output = gr.Textbox(label="Source Path")
            file_button = gr.Button("Process", variant="primary")

            file_input.change(fn=update_file_preview, inputs=file_input, outputs=file_preview)

            def process_file_fn(p, f):
                if not p:
                    return "## Set API key first", ""
                return p.process_file(f)

            file_button.click(
                fn=process_file_fn,
                inputs=[processor_state, file_input],
                outputs=[file_output, file_raw_output]
            )

        # URL Tab
        with gr.Tab("URL Input"):
            with gr.Row():
                url_input = gr.Textbox(label="URL to Image/PDF")
            url_preview = gr.Gallery(label="Preview", elem_classes="preview")
            url_output = gr.Markdown(label="OCR Result", elem_classes="output-markdown")
            url_raw_output = gr.Textbox(label="Source URL")
            url_button = gr.Button("Process", variant="primary")

            url_input.change(fn=update_url_preview, inputs=url_input, outputs=url_preview)

            def process_url_fn(p, u):
                if not p:
                    return "## Set API key first", ""
                return p.process_url(u)

            url_button.click(
                fn=process_url_fn,
                inputs=[processor_state, url_input],
                outputs=[url_output, url_raw_output]
            )

        gr.Examples(
            examples=[],
            inputs=[file_input, url_input]
        )
    return demo

if __name__ == "__main__":
    os.environ['START_TIME'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== Application Startup at {os.environ['START_TIME']} ===")
    demo = create_interface()
    demo.launch(share=True, max_threads=1)