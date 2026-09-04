"""
Launch the review app:  python -m sem_review_app
"""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sem_review_app",
        description="Review and finalise a pre-analysed folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--folder", help="Open this analysis folder on startup")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    import gradio as gr

    from .ui import APP_CSS, create_interface

    if args.folder:
        from . import loading

        print(loading.load_folder(args.folder)[0])

    create_interface().launch(
        share=args.share, server_name=args.host, server_port=args.port,
        css=APP_CSS, theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
