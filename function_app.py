import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import logging
import azure.functions as func
from azure.cosmos.exceptions import CosmosHttpResponseError

from engines.hybrid_engine import HybridRecommendationEngine
import azure_helpers.data_loading as db

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

engine = HybridRecommendationEngine(n_recs = 5)


@app.route(route="recommendations", methods=["get"])
def recommendations(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request!')
    try:
        # Try query parameters first
        user_id = req.params.get("user_id")
        article_id = req.params.get("article_id")

        # Fallback to JSON body if not provided in URL
        if not user_id or not article_id:
            try:
                req_body = req.get_json()
                user_id = user_id or req_body.get("user_id")
                article_id = article_id or req_body.get("article_id")
            except ValueError:
                pass  # ignore JSON parse errors

        # Convert to int if provided
        user_id = int(user_id) if user_id is not None else None
        article_id = int(article_id) if article_id is not None else None

        logging.info(f"user_id={user_id}, article_id={article_id}")

        try:
            recs = engine.recommend(user_id, article_id) # type: ignore
            logging.info(recs)
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
        logging.exception("Error generating recommendations")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=400
        )


@app.route(route="user_profile", methods=["GET"])
def user_profiles(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing request for user_profile endpoint.")

    # Try to extract user_id from query parameters or JSON body
    user_id = req.params.get("user_id")
    if not user_id:
        try:
            req_body = req.get_json()
            user_id = req_body.get("user_id")
        except ValueError:
            logging.warning("Invalid or missing JSON body in request.")
            return func.HttpResponse(
                "Invalid request body. Expected JSON with 'user_id'.",
                status_code=400
            )

    # Validate user_id
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        logging.warning(f"Invalid user_id provided: {user_id}")
        return func.HttpResponse(
            "Invalid 'user_id'. Must be an integer.",
            status_code=400
        )

    # Query Cosmos DB
    try:
        clicked = db.get_clicked_articles_by_user(user_id)
        logging.info(f"Successfully retrieved click history for user_id={user_id}.")
        return func.HttpResponse(
            body=json.dumps({
                "user_id": user_id,
                "clicked_articles": clicked
            }),
            status_code=200,
            mimetype="application/json"
        )

    except CosmosHttpResponseError as e:
        logging.error(f"Cosmos DB error while retrieving user {user_id}: {e}")
        return func.HttpResponse(
            "Database error occurred while retrieving user data.",
            status_code=500
        )

    except Exception as e:
        logging.exception(f"Unexpected error while processing user {user_id}: {e}")
        return func.HttpResponse(
            "Internal server error.",
            status_code=500
        )


@app.route(route="ping")
def ping(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("pong")