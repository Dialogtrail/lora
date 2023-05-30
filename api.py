from flask import Flask, stream_with_context, request, abort
from config import Config
from generate import init, generate, generate_embs
import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from utils.prompter import Prompter


def create_sentences(string):
    # return [s.strip() for s in re.split("\.|\n|\!|\?", string) if s.strip()]
    return [s.strip() for s in re.split("\n", string) if s.strip()]


def create_app(test_config=None):
    print("Initializing LLM")
    model, tokenizer, prompter, stopping_criteria = init(
        False, Config.BASE_MODEL, Config.LORA_WEIGHTS, Config.PROMPT_TEMPLATE)

    print("Initializing embedding model")
    # model = "distiluse-base-multilingual-cased-v2"
    # embedder_model = "all-mpnet-base-v2"
    embedder_model = "KBLab/sentence-bert-swedish-cased"
    embedder = SentenceTransformer(embedder_model)

    print("Initializing cross encoder")
    cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    app = Flask("api", instance_relative_config=True)
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

    @app.route('/generate', methods=['POST', 'GET'])
    def gen():
        instr = request.args.get(
            "instruction") or request.json.get("instruction") or ""
        inp = request.args.get("input") or request.json.get("input") or ""
        prompt = request.json.get("prompt", "eb")
        token = request.headers.get("x-api-token", "")
        if token != Config.SECRET:
            abort(401)

        if prompt is not None:
            prompter = Prompter(prompt)

        print("Generating")

        return stream_with_context(
            generate(
                model=model,
                tokenizer=tokenizer,
                prompter=prompter,
                stopping_criteria=stopping_criteria,
                instruction=instr,
                input=inp,
                stream_output=False,
                num_beams=1
            ))

    @app.route('/embeddings')
    def gen_embs():
        s = request.args.get("sentences", "true")
        inp = request.args.get("input", "")
        token = request.headers.get("x-api-token", "")
        if token != Config.SECRET:
            abort(401)

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

    @app.route('/test_embeddings')
    def test_embs():
        ctx = request.args.get("context", "")
        inp = request.args.get("input", "")
        token = request.headers.get("x-api-token", "")
        if token != Config.SECRET:
            abort(401)

        ctx_embs = embedder.encode(ctx)
        inp_embs = embedder.encode(inp)

        similarity = util.cos_sim(ctx_embs, inp_embs)

        return {
            "similarity": float(similarity)
        }

    return app
