if __name__ == "__main__":

    import flask
    import argparse
    from glob import glob
    import os.path
    import logging
    import subprocess
    import sys
    import os
    from pkg_resources import resource_string
    import json
    
    
    def init(args, rest):
        sconstruct = resource_string(__name__, "data/SConstruct")
        steamroller_config = resource_string(__name__, "data/steamroller_config.json.template")
        if (not args.force) and (os.path.exists("SConstruct") or os.path.exists("steamroller_config.py")):
            logging.error("Refusing to overwrite existing SConstruct or steamroller_config.json files (try \"--force\")")
        else:
            with open("SConstruct", "w") as ofd:
                ofd.write(sconstruct)
            with open("steamroller_config.json", "w") as ofd:
                ofd.write(steamroller_config)     

    def run(args, rest):
        subprocess.call(["scons"] + rest + ["CONFIG_FILE=%s" % args.config])
        
    def serve(args, rest):
        app = flask.Flask("SteamRoller")
        images = glob("work/*png")
        l = {}
        with open(args.config) as ifd:
            config = json.load(ifd)
        task_names = [x["name"] for x in config["TASKS"]]
        @app.route("/")
        def browse():

            return "<html><body><h1>{}</h1><ul>{}</ul></body></html>".format("SteamRoller results browser",
                                                                             "\n".join(["<li><a href=\"/tasks/{}\">{}</a></li>".format(t, t) for t in task_names]))

        @app.route("/tasks/<task>")
        def experiment(task):
            images = glob("work/{}*png".format(task))
            return "<html><body><h3><a href=\"/\">Back</a></h3><h1>{}</h1>{}</body></html>".format(task, "\n".join(["<img src=\"/{}\"/>".format(i) for i in images]))

        @app.route("/work/<image_file>")
        def image(image_file):
            with open(os.path.join("work", image_file)) as ifd:
                return ifd.read()
        
        app.run(port=options.port, host=options.host)

        
    parser = argparse.ArgumentParser("steamroller")
    subparsers = parser.add_subparsers(help="sub-commands")
    init_parser = subparsers.add_parser("init", help="Initialize an experiment directory")
    init_parser.add_argument("-f", "--force", dest="force", default=False, action="store_true", help="Overwrite existing files")
    init_parser.set_defaults(func=init)
    run_parser = subparsers.add_parser("run", help="Run experiments by calling SCons (additional arguments are passed through", add_help=False)
    run_parser.add_argument("-c", "--config", dest="config", default="steamroller_config.json")
    run_parser.set_defaults(func=run)
    serve_parser = subparsers.add_parser("serve", help="Serve experiment results")
    serve_parser.add_argument("-p", "--port", dest="port", default=8080, type=int)
    serve_parser.add_argument("-H", "--host", dest="host", default="localhost")
    serve_parser.add_argument("-c", "--config", dest="config", default="steamroller_config.json")
    serve_parser.set_defaults(func=serve)
    options, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    options.func(options, rest)
