from azure_helpers.blob_logger import AzureBlobLogHandler
import os, sys, logging


def get_logger(name, blob_conn_str=os.environ["AzureBlobStorageConnectionString"]):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    # Stream handler (Log Stream)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console)

    # Blob handler
    blob_conn = blob_conn_str   # same storage as function
    handler = AzureBlobLogHandler(
        conn_str=blob_conn,
        container_name="azure-bookrec-models-blob",
        blob_prefix="app"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    logger.info("Blob logging initialized.")
    return logger
