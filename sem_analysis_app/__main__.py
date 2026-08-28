"""
Launch the web application:  python -m sem_analysis_app
"""

import argparse

from pathlib import Path

from .ui import APP_CSS, create_interface

# The Scale tab draws on a canvas and needs its own client code; Gradio
# injects it through launch(head=...).
SCALE_JS = (Path(__file__).parent / "static" / "scale_canvas.js").read_text()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sem_analysis_app",
                                     description="SEM particle analysis web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Expose a public Gradio link")
    args = parser.parse_args(argv)

    import gradio as gr

    app = create_interface()
    # Gradio 6 takes theme and css at launch, not on Blocks.
    app.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        css=APP_CSS,
        theme=gr.themes.Soft(),
        head=f"<script>{SCALE_JS}</script>",
    )


if __name__ == "__main__":
    main()
