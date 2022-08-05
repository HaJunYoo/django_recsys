from django.urls import path
from . import views

app_name = "predict"

urlpatterns = [
    path('', views.predict, name='prediction_page'),
    path('img-predict/', views.img_predict, name='image_prediction_page'),
    # path('item-predict/', views.item_predict, name='item_prediction_page'),

    path('topics/', views.view_topic, name='topics'),

    path('results/', views.view_results, name='results'),
    path('wordcloud/', views.view_wordcloud, name='wordcloud'),
]