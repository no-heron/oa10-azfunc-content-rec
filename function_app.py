import json
import logging
import azure.functions as func
from engines.hybrid_engine import HybridRecommendationEngine

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