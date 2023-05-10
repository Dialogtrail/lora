from flask import Flask, stream_with_context, request, abort
from config import Config
from generate import init, generate, generate_embs


def create_app(test_config=None):
    print("Initializing")
    model, tokenizer, prompter, stopping_criteria = init(
        False, Config.BASE_MODEL, Config.LORA_WEIGHTS, Config.PROMPT_TEMPLATE)

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

        print("Embeddings")
        embs = generate_embs(model, tokenizer, inp)
        print(embs)

        return {
            "embeddings": [float(x) for x in embs]
        }

    return app
