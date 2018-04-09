if __name__ == "__main__":

    import flask
    import argparse
    from glob import glob
    import os
    import os.path
    import logging
    import subprocess
    from pkg_resources import resource_string, resource_listdir, resource_isdir
    import json
    import re
    import gzip
    import sys
    
    def get_files(path):
        entries = resource_listdir("steamroller", path)
        files = [os.path.join(path, e) for e in entries if
                 not resource_isdir("steamroller", e)
                 and e.endswith("py")
                 and os.path.basename(e) != "resources.py"]
        dirs = [e for e in entries if resource_isdir("steamroller", e)]
        return files + sum([get_files(os.path.join(path, d)) for d in dirs], [])

    def to_texts(filenames, num):
        text = "\n".join([resource_string("steamroller", f).decode() for f in filenames])
        lines = [l for l in text.split("\n") if not re.match(r"^\s*$", l)]
        per = int(len(lines) / num)
        ret = [re.sub(r"\s", " ", "\n".join(lines[i * per:(i + 1) * per])) for i in range(num)]
        return ret

    def init(args, _):
        steamroller_config = resource_string(__name__, "static_files/steamroller_config.json").decode()
        python_comments = [re.sub(r"\s", " ", str(getattr(__builtins__, a).__doc__))
                           for a in dir(__builtins__)]
        python_code = to_texts(get_files("/"), len(python_comments))
        if not os.path.exists("tasks"):
            os.mkdir("tasks")
        with gzip.open("tasks/code_versus_comments.json.gz", "wt") as ofd:
            header = {"document_count" : len(python_comments) + len(python_code)}
            ofd.write(json.dumps(header) + "\n")
            for i, t in enumerate(python_comments):
                doc = {"_id" : str(i), "_text" : t, "_label" : "comment"}
                ofd.write(json.dumps(doc) + "\n")
            for i, t in enumerate(python_code):
                doc = {"_id" : str(len(python_comments) + i), "_text" : t, "_label" : "code"}
                ofd.write(json.dumps(doc) + "\n")

        if (not args.force) and os.path.exists("steamroller_config.json"):
            logging.error("Refusing to overwrite existing steamroller_config.json file (try \"--force\")")
        else:
            with open("steamroller_config.json", "w") as ofd:
                ofd.write(steamroller_config)

    def run(args, scons_args):
        sconstruct = resource_string(__name__, "static_files/SConstruct")
        p = subprocess.Popen(["scons"] + scons_args + ["CONFIG_FILE=%s" % args.config, "-f", "-"], stdin=subprocess.PIPE)
        p.communicate(sconstruct)

    def serve(args, _):
        app = flask.Flask("SteamRoller")
        with open(args.config) as ifd:
            config = json.load(ifd)
        task_names = [x["NAME"] for x in config["TASKS"]]
        @app.route("/")
        def browse():
            return "<html><body><h1>{}</h1><ul>{}</ul></body></html>".format("SteamRoller results browser",
                                                                             "\n".join(["<li><a href=\"/tasks/{}\">{}</a></li>".format(t, t) for t in task_names]))

        @app.route("/tasks/<task>")
        def experiment(task):
            images = glob("work/{}*png".format(task))
            return "<html><body><h3><a href=\"/\">Back</a></h3>{}</body></html>".format("\n".join(["<img src=\"/{}\"/>".format(i) for i in images]))

        @app.route("/work/<image_file>")
        def image(image_file):
            with open(os.path.join("work", image_file), "rb") as ifd:
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
    serve_parser.add_argument("--config", dest="config", default="steamroller_config.json")
    serve_parser.set_defaults(func=serve)
    options, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    options.func(options, rest)
