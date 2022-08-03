from django.urls import path
from . import views

app_name = "predict"

urlpatterns = [
    path('', views.predict, name='prediction_page'),
    path('img-predict/', views.img_predict, name='image_prediction_page'),
    path('results/', views.view_results, name='results'),
    ]