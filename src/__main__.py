if __name__ == "__main__":

    import flask
    import argparse
    from glob import glob
    import os.path
    import logging
    import subprocess
    from pkg_resources import resource_string
    
    parser = argparse.ArgumentParser("steamroller")
    parser.add_argument(dest="mode", choices=["init", "run", "serve"])
    parser.add_argument("-p", "--port", dest="port", default=8080, type=int)
    parser.add_argument("-H", "--host", dest="host", default="localhost")
    parser.add_argument("-n", "--dry_run", dest="dry_run", default=False, action="store_true")
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sconstruct = resource_string(__name__, "data/SConstruct")
    steamroller_config = resource_string(__name__, "data/steamroller_config.py.template")
    if options.mode == "init":
        if os.path.exists("SConstruct") or os.path.exists("steamroller_config.py"):
            logging.error("Refusing to overwrite existing SConstruct or steamroller_config.py files")
        else:
            with open("SConstruct", "w") as ofd:
                ofd.write(sconstruct)
            with open("steamroller_config.py", "w") as ofd:
                ofd.write(steamroller_config)     
    elif options.mode == "run":
        if options.dry_run:
            subprocess.call(["scons", "-Qn"])
        else:
            subprocess.call(["scons", "-Q"])
    elif options.mode == "serve":
        app = flask.Flask("SteamRoller")
        images = glob("work/*png")
        
        @app.route("/")
        def browse():
            return "SteamRoller results browser"

        @app.route("/experiments/<task>")
        def experiment(task):
            return task
        
        app.run(port=options.port, host=options.host)
