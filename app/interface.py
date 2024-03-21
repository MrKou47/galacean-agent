import gradio as gr
import random
import time

from lcel_version import msg_handler

gr.ChatInterface(msg_handler).launch()