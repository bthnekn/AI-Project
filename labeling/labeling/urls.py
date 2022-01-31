from django.contrib import admin
from django.urls import path, re_path, include
from django.conf.urls.static import static
from django.conf import settings
from .views import *

urlpatterns = [
	path('', Anasayfa, name='Anasayfa'),
	path('isle', isle, name='isle'),
	path('admin/', admin.site.urls)
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
