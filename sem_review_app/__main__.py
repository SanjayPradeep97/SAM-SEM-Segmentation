"""
Launch the review app:  python -m sem_review_app
"""

import argparse

# How many ports to try before giving up.
PORT_TRIES = 10


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sem_review_app",
        description="Review and finalise a pre-analysed folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870,
                        help="First port to try; the next few are used if it is "
                             "busy, unless --strict-port is given")
    parser.add_argument("--strict-port", action="store_true",
                        help="Fail rather than moving to another port")
    parser.add_argument("--folder", help="Open this analysis folder on startup")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    import gradio as gr

    from .ui import APP_CSS, create_interface

    if args.folder:
        from . import loading

        print(loading.load_folder(args.folder)[0])

    app = create_interface()
    for port in range(args.port, args.port + (1 if args.strict_port else PORT_TRIES)):
        try:
            app.launch(share=args.share, server_name=args.host,
                       server_port=port, css=APP_CSS, theme=gr.themes.Soft())
            return
        except OSError as busy:
            # Gradio asks for one port and gives up. Something else holding it is
            # ordinary — the analysis app, or a copy of this one left running —
            # and the useful response is to take the next one and say so, not to
            # print a stack trace at somebody who only wanted to open a folder.
            if "Cannot find empty port" not in str(busy):
                raise
            print(f"Port {port} is in use, trying {port + 1}...")
    raise SystemExit(
        f"Ports {args.port}-{args.port + PORT_TRIES - 1} are all in use. "
        f"Close whatever is holding them, or pass --port with a free one.")


if __name__ == "__main__":
    main()
