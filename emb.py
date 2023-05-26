from flask import Flask, stream_with_context, request, abort
from config import Config
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import re

print("Initializing")
embedder = SentenceTransformer(
    'KBLab/sentence-bert-swedish-cased')

print("Initializing cross encoder")
cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


def create_sentences(string):
    # return [s.strip() for s in re.split("\.|\n|\!|\?", string) if s.strip()]
    return [s.strip() for s in re.split("\n", string) if s.strip()]


def create_app():
    app = Flask("emb", instance_relative_config=True)
    app.config.from_object(Config)

    @app.route('/cross', methods=['POST'])
    def cross_encode():
        body = request.json
        query = body['query']
        contexts = body['contexts']
        scores = cross.predict([(query, context)
                               for context in contexts])

        result = [{"score": float(score), "context": context}
                  for score, context in zip(scores, contexts)]
        sort = sorted(result, key=lambda r: r["score"], reverse=True)

        return {
            "result": sort
        }

    @app.route('/embeddings')
    def gen_embs():
        s = request.args.get("sentences", "true")
        inp = request.args.get("input", "")

        if s == "true":
            inp = create_sentences(inp)

        embs = embedder.encode(inp)

        print(inp)
        print("----")

        res = [[float(x2) for x2 in x]
               for x in embs] if s == "true" else [float(x) for x in embs]

        return {
            "embeddings": res
        }

    return app
