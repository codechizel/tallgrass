"""Root URL configuration — admin + REST API."""

from django.contrib import admin
from django.urls import path
from legislature.api import api

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", api.urls),
]
