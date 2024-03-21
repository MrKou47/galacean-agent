# galacean-agent

## Pre-requirements

1. Python 3.7 or higher
2. Add `.env` file to the root of the project with the following content:

    ```bash
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_PROJECT="galacean agent"

    LANGCHAIN_API_KEY="your_langchain_api_key"
    OPENAI_API_KEY=your_openai_api_key
    ```

## Installation

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Launch Gradio

```bash
$ python app/interface.py
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
