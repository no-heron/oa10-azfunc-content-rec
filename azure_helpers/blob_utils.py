import logging
import os
import pickle

from azure.storage.blob import BlobServiceClient


def get_blob_service_client() -> BlobServiceClient:
    """
    Returns a connected BlobServiceClient using environment variable AzureBlobStorageConnectionString.
    """
    conn_str = os.getenv("AzureBlobStorageConnectionString")
    if not conn_str:
        raise RuntimeError("AzureBlobStorageConnectionString environment variable not set.")
    return BlobServiceClient.from_connection_string(conn_str)


def upload_file_to_blob(local_path: str, blob_name: str, container_name: str = 'azure-bookrec-models-blob'):
    """
    Upload a local file to Azure Blob Storage.
    """
    try:
        blob_service = get_blob_service_client()
        container = blob_service.get_container_client(container_name)

        # Ensure the container exists
        try:
            container.create_container()
        except Exception:
            pass  # ignore if already exists

        with open(local_path, "rb") as data:
            container.upload_blob(name=blob_name, data=data, overwrite=True)
        logging.info("Uploaded %s → %s/%s", local_path, container_name, blob_name)

    except Exception as e:
        logging.exception("Failed to upload %s: %s", local_path, e)
        raise


def download_file_from_blob(blob_name: str, local_path: str, container_name: str = 'azure-bookrec-models-blob'):
    """
    Download a blob to a local file path.
    """
    try:
        blob_service = get_blob_service_client()
        blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)

        with open(local_path, "wb") as file:
            data = blob_client.download_blob()
            file.write(data.readall())
        logging.info("Downloaded %s/%s → %s", container_name, blob_name, local_path)

    except Exception as e:
        logging.exception("Failed to download blob %s/%s: %s", container_name, blob_name, e)
        raise


def load_model_from_blob_storage(
        blob_name: str = "svdpp_model.pkl",
        container_name: str = "azure-bookrec-models-blob"
        ):
    """
    Load a pickled model directly from Azure Blob Storage into memory (no file download).
    """
    try:
        blob_service = get_blob_service_client()
        blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)

        # Stream blob data directly into memory
        stream = blob_client.download_blob()
        model_bytes = stream.readall()  # read the entire blob content
        model_obj = pickle.loads(model_bytes)

        logging.info("Loaded model '%s' from container '%s' directly into memory.", blob_name, container_name)
        return model_obj

    except Exception as e:
        logging.exception("Failed to load model '%s' from blob storage: %s", blob_name, e)
        raise