import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient

class AzureBlobLogHandler(logging.Handler):
    def __init__(self, conn_str: str, container_name: str, blob_prefix: str):
        super().__init__()
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container = self.client.get_container_client(container_name)
        self.blob_prefix = blob_prefix
        # ensure container exists
        try:
            self.container.create_container()
        except Exception:
            pass

    def emit(self, record):
        try:
            log_entry = self.format(record) + "\n"
            blob_name = f"{self.blob_prefix}/{datetime.now():%Y-%m-%d}.log"
            blob_client = self.container.get_blob_client(blob_name)

            # Ensure the blob exists, or create it as an append blob
            if not blob_client.exists():
                from azure.storage.blob import BlobType
                blob_client.create_append_blob()

            # Append the new log line
            blob_client.append_block(log_entry.encode("utf-8"))
        except Exception as e:
            print(f"Blob log write failed: {e}", flush=True)
