from flask import Flask, stream_with_context, request, abort
from config import Config
from generate import init, generate, generate_embs

from sentence_transformers import SentenceTransformer, util


def create_app(test_config=None):
    print("Initializing LLM")
    model, tokenizer, prompter, stopping_criteria = init(
        False, Config.BASE_MODEL, Config.LORA_WEIGHTS, Config.PROMPT_TEMPLATE)

    print("Initializing embedding model")
    embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    app = Flask("api", instance_relative_config=True)
    app.config.from_object(Config)

    @app.route('/generate')
    def gen():
        instr = request.args.get("instruction", "")
        inp = request.args.get("input", "")
        token = request.headers.get("x-api-token", "")
        if token != Config.SECRET:
            abort(401)

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
        inp = request.args.get("input", "")
        token = request.headers.get("x-api-token", "")
        if token != Config.SECRET:
            abort(401)

        embs = embedder.encode(inp)

        print(inp)
        print("----")

        return {
            "embeddings": [float(x) for x in embs]
        }

    @app.route('/test_embeddings')
    def gen_embs():
        ctx = request.args.get("context", "")
        inp = request.args.get("input", "")
        token = request.headers.get("x-api-token", "")
        if token != Config.SECRET:
            abort(401)

        ctx_embs = embedder.encode(ctx)
        inp_embs = embedder.encode(inp)

        similarity = util.cos_sim(ctx_embs, inp_embs)

        return {
            "similarity": similarity
        }

    return app
