                    obj = reasoning(json.loads(text))
                    if obj.get("error"):
                        status = "failed"
                        error = obj["error"]
                    for choice in obj.get("choices", []):
                        delta = choice.get("delta", {})
                        if first is None and (
                            delta.get("content") or delta.get("reasoning_content")
                        ):
                            first = time.monotonic() - started
                    if obj.get("usage"):
                        usage = obj["usage"]
                    obj["request_id"] = id
                    yield "data: " + js(obj) + "\n\n"
                if not got_done and status != "failed":
                    raise Error(
                        "上游流提前断开，未收到[DONE]", "STREAM_INCOMPLETE", 503
                    )
                yield (
                    "data: "
                    + js(
                        {
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "request_id": id,
                            "timing": timing(first, usage),
                            "usage": usage,
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as e:
                status = "failed"
                error = str(e)
                yield (
                    "data: "
                    + js(
                        {
                            "error": {
                                "code": getattr(e, "code", "STREAM_FAILED"),
                                "message": str(e),
                            },
                            "request_id": id,
                        }
                    )
                    + "\n\ndata: [DONE]\n\n"
                )
            finally:
                await asyncio.shield(finish(status, timing(first, usage), usage, error))

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": id,
            },
        )

    app.include_router(api)
    return app


def main():
    mode = os.environ.get("VERL_API_MODE", "serve")
    if mode == "rank":
        train_rank()
        return
    if mode == "worker":
        worker()
        return
    if mode == "probe":
        atomic(os.environ["VERL_API_OUTPUT"], probe())
        return
    if mode == "validate":
        from omegaconf import OmegaConf

        cfg = compose(json.loads(Path(os.environ["VERL_API_INPUT"]).read_text()))
        atomic(os.environ["VERL_API_OUTPUT"], OmegaConf.to_container(cfg, resolve=True))
        return
    parser = argparse.ArgumentParser(
        description="容器内verl API，仅三个位置参数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("service_ip", nargs="?", default="0.0.0.0", help="监听IP")
    parser.add_argument(
        "service_port", nargs="?", type=int, default=7877, help="监听端口"
    )
    parser.add_argument(
        "model_path", nargs="?", default=MODEL_DEFAULT, help="训练起点模型的容器内路径"
    )
    a = parser.parse_args()
    try:
        ipaddress.ip_address(a.service_ip)
        check(1 <= a.service_port <= 65535, "端口需1..65535")
        verify_model(a.model_path, "full")
        import uvicorn

        app = create_app(a.model_path)
        print(
            f"Verl API {VERSION}: http://{a.service_ip}:{a.service_port}/docs",
            flush=True,
        )
        print(
            f"API Key文件: {ROOT / 'api_key'}；完整规范/api-spec；训练监控/jobs/{{id}}/metrics、logs、events",
            flush=True,
        )
        uvicorn.run(app, host=a.service_ip, port=a.service_port, workers=1)
    except (Error, ValueError) as e:
        parser.exit(2, f"启动失败:{e}\n")


if __name__ == "__main__":
    main()
