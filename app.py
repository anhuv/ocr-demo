import os
import base64
import gradio as gr
from mistralai import Mistral

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
            document={
                "type": "document_url",
                "document_url": pdf_url
            }
        )
        return str(ocr_response)  # Convert response to string for display
    except Exception as e:
        return f"Error: {str(e)}"

# OCR with Uploaded PDF
def ocr_uploaded_pdf(pdf_file):
    try:
        # Upload the PDF
        uploaded_pdf = client.files.upload(
            file={
                "file_name": pdf_file.name,
                "content": open(pdf_file.name, "rb")
            },
            purpose="ocr"
        )
        # Get signed URL
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
        # Process OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            }
        )
        return str(ocr_response)
    except Exception as e:
        return f"Error: {str(e)}"

# OCR with Image URL
def ocr_image_url(image_url):
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": image_url
            }
        )
        return str(ocr_response)
    except Exception as e:
        return f"Error: {str(e)}"

# OCR with Uploaded Image
def ocr_uploaded_image(image_file):
    try:
        base64_image = encode_image(image_file.name)
        if "Error" in base64_image:
            return base64_image
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            }
        )
        return str(ocr_response)
    except Exception as e:
        return f"Error: {str(e)}"

# Document Understanding
def document_understanding(doc_url, question):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "document_url", "document_url": doc_url}
                ]
            }
        ]
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Mistral OCR & Document Understanding App") as demo:
    gr.Markdown("# Mistral OCR & Document Understanding App")
    gr.Markdown("Use this app to extract text from PDFs and images or ask questions about documents!")

    with gr.Tab("OCR with PDF URL"):
        pdf_url_input = gr.Textbox(label="PDF URL", placeholder="e.g., https://arxiv.org/pdf/2201.04234")
        pdf_url_output = gr.Textbox(label="OCR Result")
        pdf_url_button = gr.Button("Process PDF")
        pdf_url_button.click(ocr_pdf_url, inputs=pdf_url_input, outputs=pdf_url_output)

    with gr.Tab("OCR with Uploaded PDF"):
        pdf_file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_file_output = gr.Textbox(label="OCR Result")
        pdf_file_button = gr.Button("Process Uploaded PDF")
        pdf_file_button.click(ocr_uploaded_pdf, inputs=pdf_file_input, outputs=pdf_file_output)

    with gr.Tab("OCR with Image URL"):
        image_url_input = gr.Textbox(label="Image URL", placeholder="e.g., https://example.com/image.jpg")
        image_url_output = gr.Textbox(label="OCR Result")
        image_url_button = gr.Button("Process Image")
        image_url_button.click(ocr_image_url, inputs=image_url_input, outputs=image_url_output)

    with gr.Tab("OCR with Uploaded Image"):
        image_file_input = gr.File(label="Upload Image", file_types=[".jpg", ".png"])
        image_file_output = gr.Textbox(label="OCR Result")
        image_file_button = gr.Button("Process Uploaded Image")
        image_file_button.click(ocr_uploaded_image, inputs=image_file_input, outputs=image_file_output)

    with gr.Tab("Document Understanding"):
        doc_url_input = gr.Textbox(label="Document URL", placeholder="e.g., https://arxiv.org/pdf/1805.04770")
        question_input = gr.Textbox(label="Question", placeholder="e.g., What is the last sentence?")
        doc_output = gr.Textbox(label="Answer")
        doc_button = gr.Button("Ask Question")
        doc_button.click(document_understanding, inputs=[doc_url_input, question_input], outputs=doc_output)

# Launch the app
demo.launch(
    share=True,
)