"""Django URL configuration and local JSON summary view."""

import httpx
from django.http import JsonResponse
from django.urls import path
from django.views.decorators.http import require_GET

from .summary import fetch_summary

SECRET_KEY = "careops-local-summary"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
ROOT_URLCONF = __name__


@require_GET
def summary(_request):
    try:
        return JsonResponse(fetch_summary())
    except (httpx.HTTPError, TypeError, ValueError, KeyError):
        return JsonResponse({"detail": "CareOps API unavailable"}, status=502)


urlpatterns = [path("summary", summary)]
