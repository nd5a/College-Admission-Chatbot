from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.cgpitShow,name="CgpitPage"),
    path('MessageSends',views.MessageSends,name='Message_Sends'),
    path('CgpitChatbot',views.CgpitChatbot,name='CgpitChatbot')
]