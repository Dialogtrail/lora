from flask import Flask, stream_with_context, request, abort
from config import Config
from sentence_transformers import SentenceTransformer, util


def create_app(test_config=None):
    print("Initializing")

    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    app = Flask("emb", instance_relative_config=True)
    app.config.from_object(Config)

    @app.route('/test')
    def test():
        inp = request.args.get("input", "")

        print("Embeddings")
        embs = model.encode(inp)
        print(embs)

        return {
            "embeddings": [float(x) for x in embs]
        }

    return app
