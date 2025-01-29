from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def welcome_view(request):
    return JsonResponse({
        "message": "Welcome to the Fuel Route API",
        "endpoints": {
            "API Documentation": "/api/route/",
            "Calculate Route": "/api/route/ (POST)",
        },
        "usage": "Send POST requests to /api/route/ with start and end locations"
    })

urlpatterns = [
    path('', welcome_view, name='welcome'),  # Root URL
    path('admin/', admin.site.urls),
    path('api/', include('routes.urls')),
]