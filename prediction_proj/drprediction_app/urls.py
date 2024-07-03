from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("home", views.home, name="home"),
    path("evaluate", views.evaluate, name="evaluate"),
    path("result", views.get_result, name="result"),
]