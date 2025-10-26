import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import azure.functions as func
from azure.cosmos.exceptions import CosmosHttpResponseError
from opencensus.ext.azure.log_exporter import AzureLogHandler

from function_app_logging import get_logger
logger = get_logger('function-app')
logger.debug("Initializing FunctionApp instance...")

try:
    app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
    logger.info("FunctionApp instance created successfully.")
except Exception as e:
    logger.exception(f"Failed to create FunctionApp: {e}")
    raise

# Import dependencies
try:
    from engines.hybrid_engine import HybridRecommendationEngine
    logger.debug("HybridRecommendationEngine module imported successfully.")
except Exception as e:
    logger.exception(f"Failed to import HybridRecommendationEngine: {e}")
    raise

try:
    import azure_helpers.data_loading as db
    logger.debug("azure_helper.data_loading module imported successfully.")
except Exception as e:
    logger.exception(f"Failed to import azure_helper.data_loading: {e}")
    raise

# Initialize engine
try:
    engine = HybridRecommendationEngine(n_recs=5)
    logger.info("HybridRecommendationEngine initialized.")
except Exception as e:
    logger.exception("Failed to initialize HybridRecommendationEngine: {e}")
    raise


@app.route(route="ping")
def ping(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("Ping received.")


logger.debug("Initializing route recommendations.")
@app.route(route="recommendations", methods=["get"])
def recommendations(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(f'Recommendations HTTP trigger was called.')
    try:
        # Try query parameters first
        user_id = req.params.get("user_id")

        # Fallback to JSON body if not provided in URL
        if not user_id:
            try:
                req_body = req.get_json()
                user_id = user_id or req_body.get("user_id")
            except ValueError:
                pass  # ignore JSON parse errors
        # Convert to int if provided
        user_id = int(user_id) if user_id is not None else None
        logger.debug(f"user_id={user_id}")

        try:
            recs = engine.recommend(user_id) # type: ignore
            return func.HttpResponse(
                json.dumps(recs, ensure_ascii=False, indent=2),
                mimetype="application/json",
                status_code=200
            )
        except:
            return func.HttpResponse(
                "Invalid type for passed arguments. Please provide integers.",
                status_code=200
            )
    except Exception as e:
        logger.exception("Error generating recommendations")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=400
        )
logger.info("Route '/recommendations' registered.")

logger.debug("Initializing route random_users")
@app.route(route="random_users", methods=["get"])
def random_users(req: func.HttpRequest) -> func.HttpResponse:
    logger.debug(f'random_users HTTP trigger was called.')
    try:
        # Try query parameters first
        n_users = req.params.get("n_users", None)
        n_users = int(n_users) if n_users else None

        # Validate n_users
        if n_users is not None and not isinstance(n_users, int):
            return func.HttpResponse("Invalid n_users parameter", status_code=400)

        users = db.get_random_users(n_users) if n_users else db.get_random_users(10)
        return func.HttpResponse(json.dumps(users), mimetype="application/json")

    except ValueError:
        return func.HttpResponse("n_users must be an integer", status_code=400)
    except Exception as e:
        logger.error(f"Error: {e}")
        return func.HttpResponse("Internal server error", status_code=500)
logger.debug("Route '/user_profile' registered.")