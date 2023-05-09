from flask import Flask, stream_with_context, request, abort
from config import Config
from generate import init, generate


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
                stream_output=True
            ))
    return app
