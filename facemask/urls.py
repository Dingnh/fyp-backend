from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from django.conf import settings
from rest_framework import routers
from facemask.quickstart import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

router.register(r"facemasks", views.FaceMaskViewSet,
                basename="facemasks")
# router.register(r'facemask', views.FacemaskViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('admin/', admin.site.urls),
    path('facemask/detect/', views.detect, name='facemask_detect'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
