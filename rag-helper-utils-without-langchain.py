import glob
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fitz
import numpy as np
import pandas as pd
import requests
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import Image as vision_model_Image
from vertexai.vision_models import MultiModalEmbeddingModel
from rich.markdown import Markdown as rich_Markdown
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Image,
    Part,
)

# Set the environment variable to point to your service account key
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ravi/.config/gcloud/genai-434714-5b6098f8999f.json"

# Define project information
PROJECT_ID = "genai-434714"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)

text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")


text_model = GenerativeModel("gemini-1.5-pro")
multimodal_model = GenerativeModel("gemini-1.5-pro-001")
multimodal_model_flash = GenerativeModel("gemini-1.5-flash-001")


##################################################################################
# Reference:
# Gemini Safety Settings: https://ai.google.dev/gemini-api/docs/safety-settings
##################################################################################
def get_document_metadata(
    generative_multimodal_model,
    pdf_folder_path: str,
    image_save_dir: str,
    image_description_prompt: str,
    embedding_size: int = 128,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    add_sleep_after_page: bool = False,
    sleep_time_after_page: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a PDF path, an image save directory, an image description prompt, an embedding size, and a text embedding text limit as input.

    Args:
        pdf_path: The path to the PDF document.
        image_save_dir: The directory where extracted images should be saved.
        image_description_prompt: A prompt to guide Gemini for generating image descriptions.
        embedding_size: The dimensionality of the embedding vectors.
        text_emb_text_limit: The maximum number of tokens for text embedding.

    Returns:
        A tuple containing two DataFrames:
            * One DataFrame containing the extracted text metadata for each page of the PDF, including the page text, chunked text dictionaries, and chunk embedding dictionaries.
            * Another DataFrame containing the extracted image metadata for each image in the PDF, including the image path, image description, image embeddings (with and without context), and image description text embedding.
    """

    text_metadata_df_final, image_metadata_df_final = pd.DataFrame(), pd.DataFrame()


    for pdf_path in glob.glob(pdf_folder_path + "/*.pdf"):
        print(
            "\n\n",
            "Processing the file: ---------------------------------",
            pdf_path,
            "\n\n",
        )

        doc, num_pages = get_pdf_doc_object(pdf_path)

        file_name = pdf_path.split("/")[-1]
        print(f"{file_name} contains {num_pages} pages.")

        text_metadata: Dict[Union[int, str], Dict] = {}
        image_metadata: Dict[Union[int, str], Dict] = {}

        for page_num in range(num_pages):
            print(f"Processing page: {page_num + 1}")

            page = doc[page_num]

            text = page.get_text()

            (
                text,
                page_text_embedding_dict,
                chunk_text_dict,
                chunk_embeddings_dict
            ) = get_chunk_text_metadata(page)

            text_metadata[page_num] = {
                "text": text,
                "page_text_embeddings": page_text_embedding_dict,
                "chunked_text_dict": chunk_text_dict,
                "chunk_embeddings_dict": chunk_embeddings_dict
            }

            images = page.get_images()
            image_metadata[page_num] = {}
            for image_no, image in enumerate(images):
                image_number = int(image_no + 1)
                image_metadata[page_num][image_number] = {}


                image_for_gemini, image_name = get_image_for_gemini(
                    doc, image, image_no, image_save_dir, file_name, page_num
                )

                print(
                    f"Extracting image from page: {page_num + 1}, saved as: {image_name}"
                )

                if image_for_gemini is None:
                    continue

                # Generate image description using Gemini
                response = get_gemini_response(
                    generative_multimodal_model,
                    model_input=[image_description_prompt, image_for_gemini],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
                )

                image_embedding = get_image_embedding_from_multimodal_embedding_model(
                    image_uri=image_name,
                    embedding_size=embedding_size,
                )

                image_description_text_embedding = (
                    get_text_embedding_from_text_embedding_model(text=response)
                )

                image_metadata[page_num][image_number] = {
                    "img_num": image_number,
                    "img_path": image_name,
                    "img_desc": response,
                    # "mm_embedding_from_text_desc_and_img": image_embedding_with_description,
                    "mm_embedding_from_img_only": image_embedding,
                    "text_embedding_from_image_description": image_description_text_embedding,
                }

                # Add sleep to reduce issues with Quota error on API
                if add_sleep_after_page:
                    time.sleep(sleep_time_after_page)
                    print(
                        "Sleeping for ",
                        sleep_time_after_page,
                        """ sec before processing the next page to avoid quota issues. You can disable it: "add_sleep_after_page = False"  """,
                    )

    text_metadata_df = get_text_metadata_df(file_name, text_metadata)
    image_metadata_df = get_image_metadata_df(file_name, image_metadata)

    text_metadata_df_final = pd.concat(
        [text_metadata_df_final, text_metadata_df], axis=0
    )
    image_metadata_df_final = pd.concat(
        [
            image_metadata_df_final,
            image_metadata_df.drop_duplicates(subset=["img_desc"]),
        ],
        axis=0,
    )

    text_metadata_df_final = text_metadata_df_final.reset_index(drop=True)
    image_metadata_df_final = image_metadata_df_final.reset_index(drop=True)

    return text_metadata_df_final, image_metadata_df_final

def get_text_metadata_df(
    filename: str, text_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and a text metadata dictionary as input,
    iterates over the text metadata dictionary and extracts the text, chunk text,
    and chunk embeddings for each page, creates a Pandas DataFrame with the
    extracted data, and returns it.

    Args:
        filename: The filename of the document.
        text_metadata: A dictionary containing the text metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted text, chunk text, and chunk embeddings for each page.
    """

    final_data_text: List[Dict] = []

    for key, values in text_metadata.items():
        for chunk_number, chunk_text in values["chunked_text_dict"].items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["text"] = values["text"]
            data["text_embedding_page"] = values["page_text_embeddings"][
                "text_embedding"
            ]
            data["chunk_number"] = chunk_number
            data["chunk_text"] = chunk_text
            data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]

            final_data_text.append(data)

    return_df = pd.DataFrame(final_data_text)
    return_df = return_df.reset_index(drop=True)
    return return_df


def get_image_metadata_df(
    filename: str, image_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and an image metadata dictionary as input,
    iterates over the image metadata dictionary and extracts the image path,
    image description, and image embeddings for each image, creates a Pandas
    DataFrame with the extracted data, and returns it.

    Args:
        filename: The filename of the document.
        image_metadata: A dictionary containing the image metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted image path, image description, and image embeddings for each image.
    """

    final_data_image: List[Dict] = []
    for key, values in image_metadata.items():
        for _, image_values in values.items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["img_num"] = int(image_values["img_num"])
            data["img_path"] = image_values["img_path"]
            data["img_desc"] = image_values["img_desc"]
            # data["mm_embedding_from_text_desc_and_img"] = image_values[
            #     "mm_embedding_from_text_desc_and_img"
            # ]
            data["mm_embedding_from_img_only"] = image_values[
                "mm_embedding_from_img_only"
            ]
            data["text_embedding_from_image_description"] = image_values[
                "text_embedding_from_image_description"
            ]
            final_data_image.append(data)

    return_df = pd.DataFrame(final_data_image).dropna()
    return_df = return_df.reset_index(drop=True)
    return return_df



def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> list:
    """Extracts an image embedding from a multimodal embedding model.
    The function can optionally utilize contextual text to refine the embedding.

    Args:
        image_uri (str): The URI (Uniform Resource Identifier) of the image to process.
        text (Optional[str]): Optional contextual text to guide the embedding generation. Defaults to "".
        embedding_size (int): The desired dimensionality of the output embedding. Defaults to 512.
        return_array (Optional[bool]): If True, returns the embedding as a NumPy array.
        Otherwise, returns a list. Defaults to False.

    Returns:
        list: A list containing the image embedding values. If `return_array` is True, returns a NumPy array instead.
    """
    # image = Image.load_from_file(image_uri)
    image = vision_model_Image.load_from_file(image_uri)
    embeddings = multimodal_embedding_model.get_embeddings(
        image=image, contextual_text=text, dimension=embedding_size
    )  # 128, 256, 512, 1408
    image_embedding = embeddings.image_embedding

    if return_array:
        image_embedding = np.fromiter(image_embedding, dtype=float)

    return image_embedding



def get_gemini_response(
    generative_multimodal_model,
    model_input: List[str],
    stream: bool = True,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
) -> str:
    """
    This function generates text in response to a list of model inputs.

    Args:
        model_input: A list of strings representing the inputs to the model.
        stream: Whether to generate the response in a streaming fashion (returning chunks of text at a time) or all at once. Defaults to False.

    Returns:
        The generated text as a string.
    """
    response = generative_multimodal_model.generate_content(
        model_input,
        generation_config=generation_config,
        stream=stream,
        safety_settings=safety_settings,
    )
    response_list = []

    for chunk in response:
        try:
            response_list.append(chunk.text)
        except Exception as e:
            print(
                "Exception occurred while calling gemini. Something is wrong. Lower the safety thresholds [safety_settings: BLOCK_LOW_AND_ABOVE ] if not already done. -----",
                e,
            )
            response_list.append("Exception occurred")
            continue
    response = "".join(response_list)

    return response

def get_image_for_gemini(
    doc: fitz.Document,
    image: tuple,
    image_no: int,
    image_save_dir: str,
    file_name: str,
    page_num: int,
) -> Tuple[Image, str]:
    """
    Extracts an image from a PDF document, converts it to JPEG format (handling color conversions), saves it, and loads it as a PIL Image Object.
    """

    try:
        xref = image[0]
        pix = fitz.Pixmap(doc, xref)

        # Check and convert color space if needed
        if pix.colorspace not in (fitz.csGRAY, fitz.csRGB, fitz.csCMYK):
            pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB

        # Now save as JPEG
        image_name = f"{image_save_dir}/{file_name}_image_{page_num}_{image_no}_{xref}.jpeg"
        os.makedirs(image_save_dir, exist_ok=True)
        pix.save(image_name)

        image_for_gemini = Image.load_from_file(image_name)  # Use Image.open for loading
        return image_for_gemini, image_name

    except Exception as e:  # Catch-all for unexpected errors
        print(f"Unexpected error processing image: {e}")
        return None, None



def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
) -> list:
    """
    Generates a numerical text embedding from a provided text input using a text embedding model.

    Args:
        text: The input text string to be embedded.
        return_array: If True, returns the embedding as a NumPy array.
                      If False, returns the embedding as a list. (Default: False)

    Returns:
        list or numpy.ndarray: A 768-dimensional vector representation of the input text.
                               The format (list or NumPy array) depends on the
                               value of the 'return_array' parameter.
    """
    embeddings = text_embedding_model.get_embeddings([text])

    text_embedding = [embedding.values for embedding in embeddings][0]              # Debug [0]

    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # returns 768 dimensional array
    return text_embedding

def get_page_text_embedding(text_data: Union[dict, str]) -> dict:
    """
    * Generates embeddings for each text chunk using a specified embedding model.
    * Takes a dictionary of text chunks and an embedding size as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding embeddings.

    Args:
        text_data: Either a dictionary of pre-chunked text or the entire page text.
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A dictionary where keys are chunk numbers or "text_embedding" and values are the corresponding embeddings.

    """
    embeddings_dict = {}

    if not text_data:
        return embeddings_dict

    if isinstance(text_data, dict):
        for chunk_number, chunk_value in text_data.items():
            text_embed = get_text_embedding_from_text_embedding_model(text=chunk_value)
            embeddings_dict[chunk_number] = text_embed
    else:
        # Process the first 1000 characters of the page text
        text_embed = get_text_embedding_from_text_embedding_model(text=text_data)
        embeddings_dict["text_embedding"] = text_embed

    return embeddings_dict


def get_text_overlapping_chunk(
    text: str, character_limit: int = 1000, overlap: int = 100
) -> dict:
    """
    * Breaks a text document into chunks of a specified size, with an overlap between chunks to preserve context.
    * Takes a text document, character limit per chunk, and overlap between chunks as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding text chunks.

    Args:
        text: The text document to be chunked.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).

    Returns:
        A dictionary where keys are chunk numbers and values are the corresponding text chunks.

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Initialize variables
    chunk_number = 1
    chunked_text_dict = {}

    # Iterate over text with the given limit and overlap
    for i in range(0, len(text), character_limit - overlap):
        end_index = min(i + character_limit, len(text))
        chunk = text[i:end_index]

        # Encode and decode for consistent encoding
        chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode(
            "utf-8", "ignore"
        )

        # Increment chunk number
        chunk_number += 1

    return chunked_text_dict


def get_chunk_text_metadata(
    page: fitz.Page,
    character_limit: int = 100,
    overlap: int = 10,
    embedding_size: int = 128,
) -> tuple[str, dict, dict, dict]:
    """
    * Extracts text from a given page object, chunks it, and generates embeddings for each chunk.
    * Takes a page object, character limit per chunk, overlap between chunks, and embedding size as input.
    * Returns the extracted text, the chunked text dictionary, and the chunk embeddings dictionary.

    Args:
        page: The fitz.Page object to process.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A tuple containing:
            - Extracted page text as a string.
            - Dictionary of embeddings for the entire page text (key="text_embedding").
            - Dictionary of chunked text (key=chunk number, value=text chunk).
            - Dictionary of embeddings for each chunk (key=chunk number, value=embedding).

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Extract text from the page
    text: str = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")

    # Get whole-page text embeddings
    page_text_embeddings_dict: dict = get_page_text_embedding(text)

    # Chunk the text with the given limit and overlap
    chunked_text_dict: dict = get_text_overlapping_chunk(text, character_limit, overlap)

    # Get embeddings for the chunks
    chunk_embeddings_dict: dict = get_page_text_embedding(chunked_text_dict)

    return text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict




def get_pdf_doc_object(pdf_path: str) -> tuple[fitz.Document, int]:
    """
    Opens a PDF file using fitz.open() and returns the PDF document object and the number of pages.

    :param
        pdf_path: The path to the PDF file.

    :return:
        A tuple containing the 'fitz.Document' object and the number of pages in the PDF.

    :raises
        FileNotFoundError: If the provided PDF path is invalid.
    """

    # Open the PDF file.
    doc: fitz.Document =  fitz.open(pdf_path)

    # Get the number of pages in the PDF file.
    num_pages: int = len(doc)

    return doc, num_pages


if __name__ == "__main__":
    image_description_prompt = """Explain what is going on in the image.
    If it's a table, extract all elements of the table.
    If it's a graph, explain the findings in the graph.
    Do not include any numbers that are not mentioned in the image.
    """

    get_document_metadata(
        generative_multimodal_model=multimodal_model,
        pdf_folder_path="./data",
        image_save_dir="./images",
        image_description_prompt=image_description_prompt
    )

    print("Done")
