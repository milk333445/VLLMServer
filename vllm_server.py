import argparse
import asyncio
import json
from contextlib import asynccontextmanager
import os
import importlib
import inspect
import yaml
import ssl
import yaml
from pydantic import BaseModel
from typing import List, Union, Optional
import numpy as np
import asyncio
from enum import Enum
import contextvars
import functools

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request, HTTPException, Header
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import add_global_metrics_labels
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest, ErrorResponse
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

from embedding_engine.generator import EmbedRerankBuilder


load_models = {}

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] # input text or list of input texts
    model: Optional[str] = None # model name
    encoding_format: Optional[str] = "float" # encoding format, default is float
    query: Optional[str] = None # query string


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log(engine, model_name):
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()
            
    for model_name, model in load_models.items():
        engine = model['engine']
        engine_args = model['engine_args']
        if not engine_args.disable_log_stats:
            asyncio.create_task(_force_log(engine, model_name))

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the header."
    )
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server using @app.middleware('http'). "
        "If a class is provided, vLLM will add it to the server using app.add_middleware(). "
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


app.add_middleware(MetricsMiddleware)  # Trace HTTP server metrics
app.add_route("/metrics", metrics)  # Exposes HTTP metrics


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    model_name = list(load_models.keys())[0]
    err = load_models[model_name]['chat_serving'].create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    for model_name, model in load_models.items():
        await model['engine'].check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = list(load_models.keys())
    return JSONResponse(content={"models": models})


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    model_name = request.model
    logger.info(f"model_name: {model_name}")
    if model_name not in load_models or model_name == None:
        return JSONResponse(
            content={"error": f"Model {model_name} not found"},
            status_code=400
        )
    
    generator = await load_models[model_name]['chat_serving'].create_chat_completion(
        request, raw_request
    )

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    model_name = request.model
    if model_name not in load_models or model_name == None:
        return JSONResponse(
            content={"error": f"Model {model_name} not found"},
            status_code=400
        )
    
    generator = await load_models[model_name]['completion_serving'].create_completion(
        request, raw_request
    )
    
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    model_name = request.model
    
    if request.query is not None:
        if model_name not in builder.reranking_model_configs:
            raise HTTPException(status_code=400, detail=f"Re-ranking model {model_name} not found")
    else:
        if model_name not in builder.embedding_model_configs:
            raise HTTPException(status_code=400, detail=f"Embedding model {model_name} not found")
        
    if isinstance(request.input, str):
        inputs = [request.input]
    else:  
        inputs = request.input
        
    try:
        model = getattr(builder, model_name)
        response_data = []
        
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        
        if request.query is not None:
            func_call = functools.partial(ctx.run, model.rerank, request.query, inputs) # 等於 model.rerank(request.query, inputs)
            scores = await loop.run_in_executor(None, func_call)

            for idx, score in enumerate(scores):
                response_data.append(
                    {
                        "object":"reranking",
                        "embedding":float(score),
                        "index":idx
                    }
                )
                
            return {
                "object":"list",
                "data":response_data,
                "model":model_name,
                "usage": {
                    "prompt_tokens": len(request.query.split()),
                    "total_tokens": sum(len(text.split()) for text in inputs)
                }
            }
        
        else:
            func_call = functools.partial(ctx.run, model.get_embeddings, inputs)
            embeddings = await loop.run_in_executor(None, func_call)
        
            for idx, embedding in enumerate(embeddings):
                response_data.append(
                    {
                        "object":"embedding",
                        "embedding":embedding.tolist(),
                        "index":idx
                    }
                )
            
            return {
                "object":"list",
                "data":response_data,
                "model":model_name,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in inputs),
                    "total_tokens": sum(len(text.split()) for text in inputs)
                }
            }
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        raise FileNotFoundError(f"Config file {config_path} not found")


def update_args_with_config(args, config):
    server_config = config.get("server", {})
    for key, value in server_config.items():
        if hasattr(args, key):
            setattr(args, key, value)


def load_multiple_models(config, args):
    model_config = config.get("LLM_engines", {})
    for model_name, model_params in model_config.items():
        try:
            logger.info(f"Loading model {model_name} with args {model_params}")
            # Load engine
            engine_args = AsyncEngineArgs(
                model=model_params.get("model"),
                tokenizer=model_params.get("tokenizer", model_params.get("model")),
                dtype=model_params.get("dtype", "float16"),
                max_model_len=model_params.get("max_model_len"),
                tensor_parallel_size=model_params.get("tensor_parallel_size", 1),
                gpu_memory_utilization=model_params.get("gpu_memory_utilization", 0.9),
                max_num_seqs=model_params.get("max_num_seqs", 10),
                quantization=model_params.get("quantization", None),
            )

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            engine = AsyncLLMEngine.from_engine_args(
                engine_args,
            )

            add_global_metrics_labels(model_name=engine_args.model)
        
            load_models[model_name] = {
                "engine": engine,
                "engine_args": engine_args,
                "chat_serving":OpenAIServingChat(engine, model_name, "assistant", None),
                "completion_serving":OpenAIServingCompletion(engine, model_name)
            }
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")


if __name__ == "__main__":
    """
    example.yaml:
        server:
            host: "0.0.0.0" 
            port: 8947
            uvicorn_log_level: "info"
            cuda: "1"

        LLM_engines:
            Qwen2-7B-Instruct:
                model: "Qwen2-7B-Instruct"
                tokenizer: "Qwen2-7B-Instruct"
                dtype: "float16"
                tensor_parallel_size: 1
                gpu_memory_utilization: 0.25
                max_num_seqs: 20
                max_model_len: 16000
            Qwen1.5-14B-Chat:
                model: "Qwen1.5-14B-Chat"
                tokenizer: "Qwen1.5-14B-Chat"
                dtype: "float16"
                tensor_parallel_size: 1
                gpu_memory_utilization: 0.55
                max_num_seqs: 20
                max_model_len: 17072

        embedding_models:
            m3e-base:
                model_name: "moka-ai/m3e-base"
                model_path: "./embedding_engine/model/embedding_model/m3e-base-model"
                tokenizer_path: "./embedding_engine/model/embedding_model/m3e-base-tokenizer"
                max_length: 512
                use_gpu: True
                use_float16: True
        reranking_models:
            bge-reranker-large:
                model_name: 'BAAI/bge-reranker-large'
                model_path: "./embedding_engine/model/reranking_model/bge-reranker-large-model"
                tokenizer_path: "./embedding_engine/model/reranking_model/bge-reranker-large-tokenizer"  
                max_length: 512
                use_gpu: True  
                use_float16: True
    """
    
    args = parse_args()
    if args.config:
        config = load_config(args.config)
        server_config = config.get("server", {})
        cuda_config = server_config.get('cuda', "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_config
        builder = EmbedRerankBuilder(config_path=args.config, logger=logger)
        load_multiple_models(config, args)
        update_args_with_config(args, config) 

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    logger.info(f"args: {args}")

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
