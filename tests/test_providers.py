import importlib.util
import json
import os
import unittest
from unittest.mock import patch

HTTPX = importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(HTTPX, "Provider tests require the providers extra")
class ProviderTests(unittest.TestCase):
    def test_ollama_uses_loopback_and_validates_choice_schema(self):
        import httpx

        from careops.providers.ollama import OllamaGenerator

        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(str(request.url), "http://127.0.0.1:11512/api/chat")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["format"]["properties"]["choice"]["enum"], ["A", "X"])
            return httpx.Response(200, json={
                "done": True, "done_reason": "stop", "message": {"content": '{"choice":"A"}'},
                "prompt_eval_count": 14, "eval_count": 7,
            })

        client = OllamaGenerator("my-qwen", port=11512, transport=httpx.MockTransport(handler))
        result = client.complete("select", "fixture", ("A", "X"))
        self.assertEqual((result.value, result.input_tokens, result.output_tokens), ("A", 14, 7))

        client.transport = httpx.MockTransport(lambda _request: httpx.Response(
            200, json={"done": True, "message": {"content": '{"choice":"C"}'}},
        ))
        self.assertIsNone(client.choose("select", "fixture", ("A", "X")))

    def test_cloud_chat_uses_fixed_endpoints_and_suppresses_raw_errors(self):
        import httpx

        from careops.providers.cloud import CloudChoice
        from careops.providers.transport import ProviderFailure

        secrets = {"OPENAI_API_KEY": "dummy-key-a", "GROQ_API_KEY": "dummy-key-b"}
        with patch.dict(os.environ, secrets, clear=True):
            for provider, host, key in (("openai", "api.openai.com", "dummy-key-a"),
                                        ("groq", "api.groq.com", "dummy-key-b")):
                with self.subTest(provider=provider):
                    def handler(request):
                        self.assertEqual(request.url.host, host)
                        self.assertEqual(request.headers["Authorization"], "Bearer " + key)
                        return httpx.Response(429, json={"error": "raw response text"})
                    client = CloudChoice(provider, "demo", transport=httpx.MockTransport(handler))
                    with self.assertRaisesRegex(ProviderFailure, "^quota$"):
                        client.complete("select", "fixture", ("A", "X"))

    def test_google_scopes_distinct_credentials_and_normalizes_tokens(self):
        import httpx

        from careops.providers.cloud import CloudChoice

        credentials = {"GEMINI_API_KEY": "test-ai", "VERTEX_ACCESS_TOKEN": "test-bearer",
                       "GOOGLE_CLOUD_PROJECT": "demo-project"}
        with patch.dict(os.environ, credentials, clear=True):
            for provider in ("gemini", "vertex"):
                with self.subTest(provider=provider):
                    def handler(request):
                        body = json.loads(request.content)
                        self.assertEqual(body["contents"][0]["parts"], [{"text": "fixture"}])
                        if provider == "gemini":
                            self.assertEqual(request.url.host, "generativelanguage.googleapis.com")
                            self.assertEqual(request.headers["x-goog-api-key"], "test-ai")
                            self.assertNotIn("Authorization", request.headers)
                        else:
                            self.assertEqual(request.url.host, "aiplatform.googleapis.com")
                            self.assertIn("/projects/demo-project/locations/global/", request.url.path)
                            self.assertEqual(request.headers["Authorization"], "Bearer test-bearer")
                        return httpx.Response(200, json={
                            "candidates": [{"finishReason": "STOP",
                                            "content": {"parts": [{"text": "B"}]}}],
                            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 2,
                                              "thoughtsTokenCount": 3},
                        })
                    client = CloudChoice(provider, "demo", transport=httpx.MockTransport(handler))
                    result = client.complete("select", "fixture", ("B", "X"))
                    self.assertEqual((result.value, result.input_tokens, result.output_tokens),
                                     ("B", 20, 5))

    @unittest.skipUnless(importlib.util.find_spec("anthropic"), "Anthropic SDK required")
    def test_anthropic_sdk_uses_fixed_endpoint_and_no_retries(self):
        import httpx2

        from careops.providers.cloud import CloudChoice

        def handler(request):
            self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
            self.assertEqual(request.headers["Authorization"], "Bearer dummy-key")
            return httpx2.Response(200, json={
                "id": "msg_demo", "type": "message", "role": "assistant", "model": "demo",
                "content": [{"type": "text", "text": "X"}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 9, "output_tokens": 1},
            })

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy-key"}, clear=True):
            client = CloudChoice("anthropic", "demo", transport=httpx2.MockTransport(handler))
            result = client.complete("select", "fixture", ("A", "X"))
        self.assertEqual((result.value, result.input_tokens, result.output_tokens), ("X", 9, 1))

    def test_policy_runtime_loader_accepts_only_local_engines(self):
        from careops.answer import load_answerer

        with self.assertRaisesRegex(ValueError, "transformers or ollama"):
            load_answerer(engine="gemini")

    def test_cloud_smoke_stops_after_one_transport_failure(self):
        from careops.providers import smoke
        from careops.providers.transport import ProviderFailure

        with patch.object(smoke, "CloudChoice") as client:
            client.return_value.complete.side_effect = ProviderFailure("quota")
            report = smoke.run("groq", "demo")
        self.assertEqual(report["request_attempts"], 1)
        self.assertEqual(report["cases"][0]["error"], "quota")
